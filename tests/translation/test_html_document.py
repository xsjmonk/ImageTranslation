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

    def test_attribute_spelling_preserved(self):
        """Valid input: attribute spelling (quoting, case) round-trips
        exactly via the lexical raw-tag layer."""
        html = "<img src='pic.jpg'>"
        doc = HTMLDocument(html)
        assert html == doc.serialize()

    def test_void_elements(self):
        html = "<p>line1<br>line2<img src='a.png'>end</p>"
        doc = HTMLDocument(html)
        assert doc.serialize() == html

    def test_self_closing_spelling_preserved(self):
        """Exact lexical preservation: <br/> stays <br/>, <br> stays <br>."""
        assert HTMLDocument("<br/>").serialize() == "<br/>"
        assert HTMLDocument("<br>").serialize() == "<br>"

    def test_malformed_markup_canonicalized(self):
        """Normalization boundary: malformed input (implied/stray tags)
        falls back to the parser's canonical serialization."""
        html = "<p>a<span>b<p>c</p>"  # implied </span>, </p>
        doc = HTMLDocument(html)
        assert doc.serialize() == "<p>a<span>b</span></p><p>c</p>"

    def test_comments_preserved(self):
        html = "<!-- 中文注释 --><p>text</p>"
        doc = HTMLDocument(html)
        assert doc.serialize() == html

    def test_doctype_preserved(self):
        html = "<!DOCTYPE html><html><body>x</body></html>"
        doc = HTMLDocument(html)
        assert doc.serialize().startswith("<!DOCTYPE html>")

    def test_entities_preserved_exact_spelling(self):
        """Lexical preservation: character references round-trip in their
        EXACT source spelling — never decoded, never normalized."""
        html = "<p>a &amp; b &lt;c&gt; &#233; &nbsp; &#xA0;</p>"
        doc = HTMLDocument(html)
        assert doc.serialize() == html
        # no decode happened anywhere in the tree
        for node in doc.text_nodes():
            assert "é" not in node.text
            assert "\xa0" not in node.text

    def test_entity_never_becomes_markup(self):
        """&lt;br&gt; is literal text; it never produces a <br> element."""
        doc = HTMLDocument("<p>中文&lt;br&gt;English</p>")
        assert [e.tag for e in doc.element_nodes() if e.tag != "#document"] == ["p"]
        assert doc.serialize() == "<p>中文&lt;br&gt;English</p>"

    def test_quoted_gt_inside_attribute(self):
        """A '>' inside a quoted attribute value is not a tag terminator."""
        html = '<p title="a > b">中文</p>'
        doc = HTMLDocument(html)
        assert doc.serialize() == html
        p = [e for e in doc.element_nodes() if e.tag == "p"][0]
        assert p.attrs == [("title", "a > b")]

    def test_repeated_identical_entities(self):
        """Every occurrence of the same entity keeps its own marker and is
        restored exactly — no deduplication, no count loss."""
        html = "<p>&nbsp;中文&nbsp;English&nbsp;结尾&nbsp;</p>"
        doc = HTMLDocument(html)
        assert doc.serialize() == html
        assert doc.serialize().count("&nbsp;") == 4
        # markers are distinct per occurrence
        markers = {n.text for n in doc.text_nodes() if "\x02" in n.text}
        assert len(markers) == 1  # one text node holds all four markers

    def test_whitespace_sensitive_attributes(self):
        """Attribute whitespace is preserved exactly via the raw tag
        spelling (valid input)."""
        html = '<p class="a  b" data-x=" spaced ">中文</p>'
        doc = HTMLDocument(html)
        assert doc.serialize() == html

    def test_entities_in_attributes_preserved(self):
        """Entity spellings inside attribute values round-trip exactly via
        the raw tag spelling (valid input)."""
        html = '<a href="/x?a=1&amp;b=2" title="a&#160;b">链接</a>'
        doc = HTMLDocument(html)
        assert doc.serialize() == html

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
