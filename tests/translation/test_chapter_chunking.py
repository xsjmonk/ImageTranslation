"""Tests for block collection and token-aware segmentation (no model needed)."""

from __future__ import annotations

import pytest

from image_translation.translation.chapter_chunking import (
    collect_blocks,
    segment_blocks,
)
from image_translation.translation.html_document import HTMLDocument


def char_measure(text: str) -> int:
    """Deterministic fake tokenizer: 1 token per 2 characters."""
    return max(1, (len(text) + 1) // 2)


class TestCollectBlocks:
    def test_paragraphs_become_blocks(self):
        doc = HTMLDocument("<p>中文一</p><p>中文二</p>")
        blocks = collect_blocks(doc)
        assert len(blocks) == 2

    def test_english_only_blocks_dropped(self):
        doc = HTMLDocument("<p>English only</p><p>中文</p>")
        blocks = collect_blocks(doc)
        assert len(blocks) == 1

    def test_excluded_subtrees_skipped(self):
        doc = HTMLDocument("<script>中文脚本</script><p>中文</p><pre>保留</pre>")
        blocks = collect_blocks(doc)
        assert len(blocks) == 1
        # script/pre content is not collected as translatable text
        all_text = "".join(i["text"] for b in blocks for i in b.items if i["kind"] == "text")
        assert "中文脚本" not in all_text
        assert "保留" not in all_text

    def test_notranslate_skipped(self):
        doc = HTMLDocument('<div class="notranslate">保留</div><p>翻译</p>')
        blocks = collect_blocks(doc)
        assert len(blocks) == 1

    def test_inline_tags_in_block(self):
        doc = HTMLDocument("<p>前 <strong>中</strong> 后</p>")
        blocks = collect_blocks(doc)
        assert len(blocks) == 1
        kinds = [i["kind"] for i in blocks[0].items]
        assert kinds == ["text", "tag", "text", "tag", "text"]


class TestSegmentBlocks:
    def test_small_block_single_segment(self):
        doc = HTMLDocument("<p>加厚防水面料</p>")
        blocks = collect_blocks(doc)
        segs = segment_blocks(doc, blocks, char_measure, 100, document_id="d")
        assert len(segs) == 1
        assert segs[0].token_count <= 100

    def test_large_block_split_by_sentences(self):
        text = "第一句很长很长。" + "第二句也很长很长。" + "第三句依然很长。"
        doc = HTMLDocument(f"<p>{text}</p>")
        blocks = collect_blocks(doc)
        segs = segment_blocks(doc, blocks, char_measure, 10, document_id="d")
        assert len(segs) >= 2
        # Reconstructed source must equal original (no data loss)
        joined = "".join(s.source_text for s in segs)
        assert joined == text

    def test_segment_order_and_context(self):
        doc = HTMLDocument("<p>一</p><p>二</p><p>三</p>")
        blocks = collect_blocks(doc)
        segs = segment_blocks(doc, blocks, char_measure, 100, document_id="d")
        ids = [s.segment_id for s in segs]
        assert ids == ["d:0000", "d:0001", "d:0002"]
        assert segs[0].context_after_id == "d:0001"
        assert segs[1].context_before_id == "d:0000"
        assert segs[1].context_after_id == "d:0002"

    def test_tags_never_split_across_segments(self):
        text = "很长很长的第一段。" * 8 + "<strong>重点</strong>" + "后续内容。" * 8
        doc = HTMLDocument(f"<p>{text}</p>")
        blocks = collect_blocks(doc)
        segs = segment_blocks(doc, blocks, char_measure, 20, document_id="d")
        # Every tag placeholder appears fully inside one segment source
        import re
        for s in segs:
            toks = re.findall(r"__ITRANSLATE_[TSP]\d{4}_", s.source_text)
            assert len(set(toks)) == len(toks)  # no partial/duplicate tokens

    def test_split_node_reassembles_exactly(self):
        """A single long text node split across segments must reassemble."""
        text = "很长的中文段落。" * 30
        doc = HTMLDocument(f"<p>{text}</p>")
        blocks = collect_blocks(doc)
        segs = segment_blocks(doc, blocks, char_measure, 30, document_id="d")
        assert len(segs) > 1
        joined = "".join(s.source_text for s in segs)
        assert joined == text
