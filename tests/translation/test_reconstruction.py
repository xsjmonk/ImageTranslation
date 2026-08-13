"""Tests for reconstruction: ordered merge, fingerprint, excluded content."""

from __future__ import annotations

import pytest

from image_translation.translation.chapter_chunking import (
    collect_blocks,
    segment_blocks,
)
from image_translation.translation.exceptions import StructuredTranslationError
from image_translation.translation.html_document import HTMLDocument
from image_translation.translation.reconstruction import rebuild_document


def char_measure(text: str) -> int:
    return max(1, (len(text) + 1) // 2)


def _translate_segments(segs):
    """Simulate the model: wrap each CJK run with 'EN:', preserving placeholders."""
    import re
    for s in segs:
        s.translated_text = re.sub(
            r"[\u4e00-\u9fff]+", lambda m: "EN:" + m.group(0), s.source_text
        )


class TestRebuild:
    def test_basic_merge(self):
        doc = HTMLDocument("<p>加厚防水面料</p>")
        fp = doc.fingerprint()
        blocks = collect_blocks(doc)
        segs = segment_blocks(doc, blocks, char_measure, 100, document_id="d")
        _translate_segments(segs)
        out = rebuild_document(doc, segs, fp)
        assert out == "<p>EN:加厚防水面料</p>"

    def test_inline_tags_preserved(self):
        doc = HTMLDocument("<p>前 <strong>中</strong> 后</p>")
        fp = doc.fingerprint()
        blocks = collect_blocks(doc)
        segs = segment_blocks(doc, blocks, char_measure, 100, document_id="d")
        _translate_segments(segs)
        out = rebuild_document(doc, segs, fp)
        assert out == "<p>EN:前 <strong>EN:中</strong> EN:后</p>"

    def test_english_only_untouched(self):
        doc = HTMLDocument("<p>English stays</p><p>中文</p>")
        fp = doc.fingerprint()
        blocks = collect_blocks(doc)
        segs = segment_blocks(doc, blocks, char_measure, 100, document_id="d")
        assert len(segs) == 1
        _translate_segments(segs)
        out = rebuild_document(doc, segs, fp)
        assert "English stays" in out
        assert "EN:中文" in out

    def test_excluded_content_unchanged(self):
        doc = HTMLDocument('<script>var a="中文"</script><p>中文</p>')
        fp = doc.fingerprint()
        blocks = collect_blocks(doc)
        segs = segment_blocks(doc, blocks, char_measure, 100, document_id="d")
        _translate_segments(segs)
        out = rebuild_document(doc, segs, fp)
        assert 'var a="中文"' in out

    def test_fingerprint_mismatch_fails_closed(self):
        doc = HTMLDocument("<p>中文</p>")
        fp = doc.fingerprint()
        blocks = collect_blocks(doc)
        segs = segment_blocks(doc, blocks, char_measure, 100, document_id="d")
        _translate_segments(segs)
        # Break the structure after the fact
        doc.root.children.append(HTMLDocument("<p>x</p>").root.children[0])
        with pytest.raises(StructuredTranslationError, match="fingerprint"):
            rebuild_document(doc, segs, fp)

    def test_piece_count_mismatch_fails(self):
        doc = HTMLDocument("<p>前 <strong>中</strong> 后</p>")
        fp = doc.fingerprint()
        blocks = collect_blocks(doc)
        segs = segment_blocks(doc, blocks, char_measure, 100, document_id="d")
        # Model dropped the tag placeholders -> piece count != run count
        segs[0].translated_text = "model output without tags"
        with pytest.raises(StructuredTranslationError, match="piece count"):
            rebuild_document(doc, segs, fp)

    def test_split_node_reassembled(self):
        import re
        text = "很长的中文段落。" * 30
        doc = HTMLDocument(f"<p>{text}</p>")
        fp = doc.fingerprint()
        blocks = collect_blocks(doc)
        segs = segment_blocks(doc, blocks, char_measure, 30, document_id="d")
        _translate_segments(segs)
        out = rebuild_document(doc, segs, fp)
        # Contract: per-segment translations concatenate in segment order.
        # (Hard splits may land inside a CJK run; each piece is translated
        # independently, then reassembled in order — no data loss.)
        expected_inner = "".join(
            re.sub(r"[\u4e00-\u9fff]+", lambda m: "EN:" + m.group(0), s.source_text)
            for s in segs
        )
        assert out == "<p>" + expected_inner + "</p>"
        # And the EN: artifacts of the fake aside, the raw text is lossless:
        raw = re.sub(r"EN:", "", out)
        assert raw == "<p>" + text + "</p>"
