"""Tests for the lenient HTML parser/serializer/fingerprint."""

from __future__ import annotations

import pytest

from image_translation.translation.html_document import HTMLDocument


class TestRoundTrip:
    def test_nested_elements(self):
        html = "<p>Hello <strong>加厚防水面料</strong> world</p>"
        doc = HTMLDocument(html)
        assert doc.serialize() == html

    def test_attributes_round_trip(self):
        html = '<a href="https://example.com" data-id="42">中文链接</a>'
        doc = HTMLDocument(html)
        out = doc.serialize()
        assert 'href="https://example.com"' in out
        assert 'data-id="42"' in out
        assert out.startswith("<a")

    def test_attribute_quoting_normalized(self):
        """Single-quoted attributes are re-emitted with double quotes."""
        html = "<img src='pic.jpg'>"
        doc = HTMLDocument(html)
        assert '<img src="pic.jpg">' == doc.serialize()

    def test_void_elements(self):
        html = "<p>line1<br>line2<img src='a.png'>end</p>"
        doc = HTMLDocument(html)
        out = doc.serialize()
        assert "<br>" in out
        assert '<img src="a.png">' in out

    def test_self_closing_normalized(self):
        """HTML5 canonicalization: <br/> and <br> both become <br>."""
        html = "<br/>"
        doc = HTMLDocument(html)
        assert doc.serialize() == "<br>"

    def test_comments_preserved(self):
        html = "<!-- 中文注释 --><p>text</p>"
        doc = HTMLDocument(html)
        assert doc.serialize() == html

    def test_doctype_preserved(self):
        html = "<!DOCTYPE html><html><body>x</body></html>"
        doc = HTMLDocument(html)
        assert doc.serialize().startswith("<!DOCTYPE html>")

    def test_entities_decoded_then_reserialized(self):
        """HTML5 parsing decodes character references; the semantic content
        (and <, >, & re-escaped on output) is preserved — not bytes."""
        html = "<p>a &amp; b &lt;c&gt; &#233;</p>"
        doc = HTMLDocument(html)
        out = doc.serialize()
        assert "&amp;" in out          # & re-escaped
        assert "&lt;c&gt;" in out      # <c> re-escaped
        assert "é" in out              # &#233; decoded to é

    def test_whitespace_and_newlines_preserved(self):
        html = "<p>  加厚\n防水 面料  </p>"
        doc = HTMLDocument(html)
        assert doc.serialize() == html

    def test_script_content_cdata(self):
        html = '<script>var x = "<b>not a tag</b>";</script><p>中文</p>'
        doc = HTMLDocument(html)
        out = doc.serialize()
        assert '<b>not a tag</b>' in out  # untouched
        assert "<p>中文</p>" in out

    def test_stray_end_tag_normalized_away(self):
        """HTML5 policy: stray end tags are ignored (deterministic)."""
        html = "<p>text</span> more</p>"
        doc = HTMLDocument(html)
        out = doc.serialize()
        assert "</span>" not in out
        assert "<p>text more</p>" == out

    def test_implied_end_tags_applied(self):
        """HTML5 policy: <p>a<div>b -> <p>a</p><div>b</div>."""
        doc = HTMLDocument("<p>a<div>b")
        assert doc.serialize() == "<p>a</p><div>b</div>"


class TestFingerprint:
    def test_fingerprint_stable_for_equivalent_docs(self):
        h1 = "<p>加厚防水面料 <strong>A</strong></p>"
        h2 = "<p>其他文本 <strong>B</strong></p>"
        d1 = HTMLDocument(h1)
        d2 = HTMLDocument(h2)
        # Same structure -> same fingerprint (text content is not hashed)
        assert d1.fingerprint() == d2.fingerprint()

    def test_fingerprint_changes_with_structure(self):
        d1 = HTMLDocument("<p>a</p>")
        d2 = HTMLDocument("<p>a</p><p>b</p>")
        assert d1.fingerprint() != d2.fingerprint()

    def test_fingerprint_changes_with_attrs(self):
        d1 = HTMLDocument('<p class="x">a</p>')
        d2 = HTMLDocument('<p class="y">a</p>')
        assert d1.fingerprint() != d2.fingerprint()

    def test_fingerprint_ignores_translated_text(self):
        """After a translation change, structure fingerprint must not move."""
        d = HTMLDocument("<p>加厚防水面料</p>")
        before = d.fingerprint()
        for node in d.text_nodes():
            if node.text.strip():
                node.text = "Thick waterproof fabric"
        assert d.fingerprint() == before


class TestExcludedSubtrees:
    def test_excluded_text_ids(self):
        html = '<script>var a=1</script><p>中文</p><code>var b=2</code>'
        doc = HTMLDocument(html)
        ids = doc.excluded_text_node_ids()
        assert len(ids) == 2  # script + code text nodes

    def test_translate_no_and_notranslate(self):
        html = '<div translate="no">保留</div><div class="notranslate">also</div><p>译</p>'
        doc = HTMLDocument(html)
        ids = doc.excluded_text_node_ids()
        assert len(ids) == 2

    def test_text_node_ids_assigned(self):
        doc = HTMLDocument("<p>a<span>b</span>c</p>")
        ids = [n.id for n in doc.text_nodes()]
        assert len(ids) == 3
        assert len(set(ids)) == 3
