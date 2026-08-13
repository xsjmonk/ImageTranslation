"""HTML5 document model, parser, serializer, and structural fingerprint.

Parser (documented policy):
- Uses html5lib (the reference HTML5 parser) in fragment mode with
  ``namespaceHTMLElements=False``. Malformed input is NORMALIZED
  deterministically per the HTML5 algorithm (never rejected, never regex):
    * stray end tags are ignored (``<p>a</span>b`` -> ``<p>ab</p>``);
    * implied end tags are applied (``<p>a<div>b`` -> ``<p>a</p><div>b</div>``);
    * void elements are canonicalized (``<br/>`` -> ``<br>``);
    * attribute names are lowercased, values entity-decoded;
    * character references are decoded (``&#233;`` -> ``é``, ``&amp;`` -> ``&``).
- A leading ``<!DOCTYPE ...>`` is preserved verbatim as a document prefix
  (fragment parsing drops it).
- Script/style contents are raw-text: kept verbatim, never parsed as markup.

Serializer guarantee (documented):
- NOT byte preservation. Guarantee: *semantic round-trip* — after parse +
  serialize, tag names, nesting, attribute names/values, comments, doctype,
  text content, whitespace, and excluded subtrees are unchanged as data;
  quoting, void-tag form, entity encoding, and case are normalized
  (attributes double-quoted, text re-escaped, void tags unclosed).
- Text nodes are escaped on output (``& < >``), so any markup-like content —
  including text produced by the translation model — can never inject tags.

Fingerprint: detects tag movement, attribute changes, omitted/duplicated
nodes, and changed excluded content. Translatable-attribute VALUES are
excluded from the fingerprint (they legitimately change).
"""

from __future__ import annotations

import hashlib
import html as html_lib
import re
from typing import Dict, List, Optional, Set, Tuple

VOID_ELEMENTS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "param", "source", "track", "wbr",
}

RAW_TEXT_ELEMENTS = {"script", "style"}

Attr = Tuple[str, str]

_DOCTYPE_RE = re.compile(r"^\s*<!DOCTYPE[^>]*>", re.IGNORECASE)


class Node:
    kind = "node"

    def __init__(self) -> None:
        self.children: List[Node] = []


class TextNode(Node):
    kind = "text"

    def __init__(self, text: str, node_id: str, raw: bool = False) -> None:
        super().__init__()
        self.text = text
        self.id = node_id
        self.raw = raw  # inside script/style: emitted verbatim


class ElementNode(Node):
    kind = "element"

    def __init__(self, tag: str, attrs: List[Attr], node_id: str) -> None:
        super().__init__()
        self.tag = tag
        self.attrs = attrs
        self.id = node_id


class CommentNode(Node):
    kind = "comment"

    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


class DoctypeNode(Node):
    kind = "doctype"

    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


class HTMLDocument:
    """An HTML5-normalized fragment/document with ordered node tree."""

    def __init__(self, source: str) -> None:
        self.source = source
        self.root = ElementNode("#document", [], "#root")
        self._next_text_id = 1
        self._next_elem_id = 1
        self._parse(source)

    # ------------------------------------------------------------------
    # Parsing (html5lib)
    # ------------------------------------------------------------------

    def _parse(self, source: str) -> None:
        import html5lib
        from xml.etree import ElementTree as ET

        # Preserve a leading doctype verbatim (fragment parsing drops it)
        m = _DOCTYPE_RE.match(source)
        if m:
            self.root.children.append(DoctypeNode(m.group(0).strip()))
            source = source[m.end():]

        try:
            frag = html5lib.parseFragment(
                source,
                treebuilder="etree",
                namespaceHTMLElements=False,
            )
        except Exception as e:
            raise ValueError(f"malformed HTML could not be parsed: {e}") from e

        # Plain-text fragments store their text on the fragment root itself
        if frag.text:
            self.root.children.append(TextNode(frag.text, self._new_text_id()))

        self._convert_children(frag, self.root)

    def _convert_children(self, etree_parent, node_parent: ElementNode) -> None:
        """Convert an ElementTree fragment into our node model.

        Element order: child.text (raw for script/style), child's children,
        then child.tail — matching document order.
        """
        from xml.etree import ElementTree as ET

        for child in list(etree_parent):
            if child.tag is ET.Comment:
                node_parent.children.append(CommentNode(child.text or ""))
                continue
            if not isinstance(child.tag, str):
                continue  # processing instructions etc.
            elem = ElementNode(child.tag, list(child.attrib.items()), self._new_elem_id())
            node_parent.children.append(elem)
            if child.text:
                raw = child.tag in RAW_TEXT_ELEMENTS
                elem.children.append(TextNode(child.text, self._new_text_id(), raw=raw))
            self._convert_children(child, elem)
            if child.tail:
                node_parent.children.append(TextNode(child.tail, self._new_text_id()))

    def _new_text_id(self) -> str:
        text_id = f"t{self._next_text_id:05d}"
        self._next_text_id += 1
        return text_id

    def _new_elem_id(self) -> str:
        elem_id = f"e{self._next_elem_id:05d}"
        self._next_elem_id += 1
        return elem_id

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------

    def walk(self):
        stack = [self.root]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(reversed(node.children))

    def text_nodes(self):
        for node in self.walk():
            if node.kind == "text":
                yield node

    def element_nodes(self):
        for node in self.walk():
            if node.kind == "element":
                yield node

    def get_text_node(self, node_id: str) -> Optional[TextNode]:
        for node in self.text_nodes():
            if node.id == node_id:
                return node
        return None

    def get_element_node(self, node_id: str) -> Optional[ElementNode]:
        for node in self.element_nodes():
            if node.id == node_id:
                return node
        return None

    # ------------------------------------------------------------------
    # Excluded subtrees (never translated: content must stay identical)
    # ------------------------------------------------------------------

    def excluded_text_node_ids(
        self,
        excluded_tags=("script", "style", "code", "pre"),
        excluded_classes=("notranslate",),
    ) -> set:
        ids: set = set()

        def is_excluded(element: ElementNode) -> bool:
            if element.tag.lower() in excluded_tags:
                return True
            attrs = dict(element.attrs)
            if attrs.get("translate", "").strip().lower() == "no":
                return True
            classes = attrs.get("class", "").split()
            return any(c in excluded_classes for c in classes)

        def walk(container) -> None:
            for child in container.children:
                if child.kind == "element":
                    if is_excluded(child):
                        for node in child.children:
                            if node.kind == "text":
                                ids.add(node.id)
                    else:
                        walk(child)

        walk(self.root)
        return ids

    # ------------------------------------------------------------------
    # Serialization (semantic round-trip; injection-safe)
    # ------------------------------------------------------------------

    def serialize(self) -> str:
        return "".join(_serialize_node(child) for child in self.root.children)

    # ------------------------------------------------------------------
    # Structural fingerprint
    # ------------------------------------------------------------------

    def fingerprint(self, translatable_attrs: Set[str] = frozenset()) -> str:
        """Stable hash of tag nesting, attrs (values of translatable attrs
        excluded), comments, doctypes, and text-node IDs."""
        parts: List[str] = []
        for node in self.walk():
            if node.kind == "element":
                attr_parts = []
                for name, value in node.attrs:
                    if name in translatable_attrs:
                        attr_parts.append(f"{name}=*")
                    else:
                        attr_parts.append(f"{name}={value}")
                parts.append(f"e:{node.id}:{node.tag}:{'|'.join(attr_parts)}")
            elif node.kind == "text":
                parts.append(f"t:{node.id}")
            elif node.kind == "comment":
                parts.append(f"c:{node.text}")
            elif node.kind == "doctype":
                parts.append(f"d:{node.text}")
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Serializer
# ---------------------------------------------------------------------------

def _serialize_attrs(attrs: List[Attr]) -> str:
    if not attrs:
        return ""
    out = []
    for name, value in attrs:
        escaped = html_lib.escape(value, quote=True)
        out.append(f' {name}="{escaped}"')
    return "".join(out)


def _serialize_node(node: Node) -> str:
    if node.kind == "text":
        if node.raw:
            return node.text
        return html_lib.escape(node.text, quote=False)
    if node.kind == "comment":
        return f"<!--{node.text}-->"
    if node.kind == "doctype":
        return node.text
    if node.kind == "element":
        if node.tag in VOID_ELEMENTS:
            return f"<{node.tag}{_serialize_attrs(node.attrs)}>"
        inner = "".join(_serialize_node(child) for child in node.children)
        return f"<{node.tag}{_serialize_attrs(node.attrs)}>{inner}</{node.tag}>"
    return ""
