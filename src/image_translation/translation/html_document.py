"""HTML5 document model, parser, serializer, and structural fingerprint.

Lexical-preservation layer (this task's guarantee):
- BEFORE html5lib parsing, a lexical scanner walks the raw source and
  replaces every character reference (``&nbsp;``, ``&#160;``, ``&#xA0;``,
  ``&amp;``, ``&lt;br&gt;``, ...) — and every bare ampersand — with a
  collision-resistant sentinel marker. The marker survives parsing as plain
  text and ``serialize()`` converts it back to the EXACT original spelling.
  Entities therefore never reach the translator as free text and are never
  normalized by the parser: ``&nbsp;``, ``&#160;`` and ``&#xA0;`` remain
  distinct, and ``&lt;br&gt;`` stays literal text — it can never become a
  real ``<br>`` element.
- The scanner also records the EXACT source spelling of every tag
  (``<br>`` vs ``<br/>``, ``</STRONG>`` vs ``</strong>``). For VALID input
  these raw spellings are attached to their elements and restored verbatim;
  for malformed input (implied/stray tags) the parser's canonical form is
  used instead (documented normalization boundary).
- Raw-text elements (script/style) are skipped entirely: their content is
  never sentinelized and stays byte-identical.

Parser (documented policy):
- Uses html5lib (the reference HTML5 parser) in fragment mode with
  ``namespaceHTMLElements=False``. Malformed input is NORMALIZED
  deterministically per the HTML5 algorithm (never rejected, never regex):
    * stray end tags are ignored (``<p>a</span>b`` -> ``<p>ab</p>``);
    * implied end tags are applied (``<p>a<div>b`` -> ``<p>a</p><div>b</div>``);
    * attribute names are lowercased and values entity-decoded in the
      PARSED TREE only; the lexical layer restores the exact source
      spelling on output (valid input).
- Script/style contents are raw-text: kept verbatim, never parsed as markup.

Serializer guarantee (documented):
- EXACT lexical preservation for all recognized inline codes in valid input:
  entity spellings (``&nbsp;``/``&#160;``/``&#xA0;``/``&amp;``/...), bare
  ampersands, and tag spellings (``<br>`` vs ``<br/>``, case, attribute
  spelling) are restored byte-for-byte from the source.
- Normalization boundary (documented, only outside the above): malformed
  markup is serialized in the parser's canonical form; attribute quoting is
  normalized when an attribute is translated (allowlist); unknown/malformed
  entity forms are covered by the bare-``&`` rule and stay literal.
- Text nodes are escaped on output (``& < >``), so any markup-like content —
  including text produced by the translation model — can never inject tags
  or entities. Model-emitted ``&nbsp;`` becomes ``&amp;nbsp;`` (text).

Fingerprint: detects tag movement, attribute changes, omitted/duplicated
nodes, and changed excluded content. Translatable-attribute VALUES are
excluded from the fingerprint (they legitimately change). Entity sentinels
are stable text markers, so the fingerprint is entity-spelling-independent.
"""

from __future__ import annotations

import hashlib
import html as html_lib
import re
import secrets
from typing import Dict, List, Optional, Set, Tuple

VOID_ELEMENTS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "param", "source", "track", "wbr",
}

RAW_TEXT_ELEMENTS = {"script", "style"}

Attr = Tuple[str, str]

_DOCTYPE_RE = re.compile(r"^\s*<!DOCTYPE[^>]*>", re.IGNORECASE)

# Character references: named (with or without the legacy trailing ';'),
# decimal, or hex. The trailing ``(?:;|(?!...))`` group keeps the ';' when
# present and still rejects prefixes of longer alphanumeric runs ("AT&T" ->
# bare '&' + "T" stays literal text).
_ENTITY_REF_RE = re.compile(
    r"&(?:#[0-9]{1,8}(?:;|(?![A-Za-z0-9]))"
    r"|#[xX][0-9a-fA-F]{1,8}(?:;|(?![A-Za-z0-9]))"
    r"|[A-Za-z][A-Za-z0-9]{0,31}(?:;|(?![A-Za-z0-9])))"
)

# Sentinel marker: STX + "ITENT" + 8-hex nonce + 4-digit index + ETX.
_ENTITY_MARKER_RE = re.compile(r"\x02ITENT([0-9a-f]{8})(\d{4})\x03")


def entity_marker(nonce: str, index: int) -> str:
    return f"\x02ITENT{nonce}{index:04d}\x03"


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
        # Exact source spellings (valid input only; None = canonical form)
        self.raw_start: Optional[str] = None
        self.raw_end: Optional[str] = None


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
    """An HTML5-normalized fragment/document with ordered node tree.

    Entity spellings and tag spellings are preserved exactly from the
    original source via the lexical layer (see module docstring).
    """

    def __init__(self, source: str) -> None:
        self.source = source
        self.root = ElementNode("#document", [], "#root")
        self._next_text_id = 1
        self._next_elem_id = 1
        self._entities: Dict[int, str] = {}
        self._nonce = secrets.token_hex(4)
        self._entity_marker_re = re.compile(
            r"\x02ITENT" + re.escape(self._nonce) + r"(\d{4})\x03"
        )
        self._parse(source)

    # ------------------------------------------------------------------
    # Parsing (html5lib, after the lexical entity/tag scan)
    # ------------------------------------------------------------------

    def _parse(self, source: str) -> None:
        import html5lib
        from xml.etree import ElementTree as ET

        # 1) Lexical scan: sentinelize character references (exact spelling
        #    preservation) and record exact raw tag spellings.
        sentinelized, self._entities, raw_tags = _lexical_scan(source, self._nonce)

        # Preserve a leading doctype verbatim (fragment parsing drops it)
        m = _DOCTYPE_RE.match(sentinelized)
        if m:
            self.root.children.append(DoctypeNode(m.group(0).strip()))
            sentinelized = sentinelized[m.end():]

        try:
            frag = html5lib.parseFragment(
                sentinelized,
                treebuilder="etree",
                namespaceHTMLElements=False,
            )
        except Exception as e:
            raise ValueError(f"malformed HTML could not be parsed: {e}") from e

        # Plain-text fragments store their text on the fragment root itself
        if frag.text:
            self.root.children.append(TextNode(frag.text, self._new_text_id()))

        self._convert_children(frag, self.root)
        self._pair_raw_tags(raw_tags)

    def _pair_raw_tags(self, raw_tags: List[Tuple[str, str]]) -> None:
        """Attach exact source tag spellings to their elements.

        For VALID input, source start tags correspond 1:1 with elements in
        document order and end tags close the element stack in order. Any
        mismatch (implied/stray tags from malformed input) invalidates the
        whole mapping: all elements fall back to the canonical serialization
        (documented normalization boundary).
        """
        if not raw_tags:
            return
        elements = [
            e for e in self.walk()
            if e.kind == "element" and e.tag != "#document"
        ]
        stack: List[ElementNode] = []
        elems = iter(elements)
        valid = True
        for name, raw in raw_tags:
            if name.startswith("/"):
                if not stack or stack[-1].tag != name[1:]:
                    valid = False
                    break
                stack.pop().raw_end = raw
                continue
            elem = next(elems, None)
            if elem is None or elem.tag != name:
                valid = False
                break
            elem.raw_start = raw
            if elem.tag not in VOID_ELEMENTS:
                stack.append(elem)
        if not valid or next(elems, None) is not None or stack:
            # Malformed input: canonical serialization for every tag.
            for e in elements:
                e.raw_start = None
                e.raw_end = None

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
    # Serialization (exact entity/tag spellings; injection-safe)
    # ------------------------------------------------------------------

    def serialize(self) -> str:
        out = "".join(_serialize_node(child) for child in self.root.children)
        # Restore entity sentinels to their EXACT source spellings. Markers
        # only exist where the lexical scanner placed them (never in model
        # output: the model never sees markers), and the per-document nonce
        # makes source collisions impossible.
        if self._entities:
            out = self._entity_marker_re.sub(
                lambda m: self._entities.get(int(m.group(1)), m.group(0)), out
            )
        return out

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
# Lexical scanner: character-reference sentinelization + raw tag spellings
# ---------------------------------------------------------------------------

_TAG_NAME_RE = re.compile(r"</?\s*([A-Za-z][A-Za-z0-9]*)")


def _lexical_scan(
    source: str, nonce: str
) -> Tuple[str, Dict[int, str], List[Tuple[str, str]]]:
    """Walk the raw source once, emitting:
    - ``sentinelized``: source with every character reference (and bare
      ampersand) replaced by a per-document sentinel marker;
    - ``entities``: marker index -> EXACT source spelling;
    - ``raw_tags``: [(name, exact_spelling), ...] for every tag in document
      order; end tags use the name with a leading ``/``.

    The scanner is context-aware: tags (``<...>`` with quoted strings),
    comments, declarations, CDATA, and script/style raw-text content are
    passed through untouched (no sentinels inside them).
    """
    entities: Dict[int, str] = {}
    raw_tags: List[Tuple[str, str]] = []
    out: List[str] = []
    i, n = 0, len(source)

    def emit_entity(raw: str) -> None:
        idx = len(entities)
        entities[idx] = raw
        out.append(entity_marker(nonce, idx))

    while i < n:
        ch = source[i]
        if ch == "<":
            if source.startswith("<!--", i):
                j = source.find("-->", i + 4)
                j = n if j < 0 else j + 3
                out.append(source[i:j])
                i = j
                continue
            if source.startswith("<![CDATA[", i):
                j = source.find("]]>", i + 9)
                j = n if j < 0 else j + 3
                out.append(source[i:j])
                i = j
                continue
            if source.startswith("<!", i) or source.startswith("<?", i):
                j = source.find(">", i)
                j = n if j < 0 else j + 1
                out.append(source[i:j])
                i = j
                continue
            # Element tag: scan to '>' honoring quoted attribute values.
            j = i + 1
            quoted = None
            while j < n:
                c = source[j]
                if quoted:
                    if c == quoted:
                        quoted = None
                elif c in "\"'":
                    quoted = c
                elif c == ">":
                    break
                j += 1
            if j >= n:
                out.append("<")  # unterminated '<': keep as plain text
                i += 1
                continue
            raw = source[i:j + 1]
            out.append(raw)
            name = ""
            tm = _TAG_NAME_RE.match(raw)
            if tm:
                name = tm.group(1).lower()
                raw_tags.append((f"/{name}" if raw.startswith("</") else name, raw))
            if name in RAW_TEXT_ELEMENTS and not raw.startswith("</"):
                # Raw-text element: skip content verbatim until its closing
                # tag (script/style content is never entity-decoded).
                close = re.search(
                    r"</\s*" + re.escape(name) + r"\s*>", source[j + 1:], re.IGNORECASE
                )
                if close:
                    end = j + 1 + close.end()
                    out.append(source[j + 1:end])
                    raw_tags.append((f"/{name}", source[j + 1 + close.start():end]))
                    i = end
                else:
                    out.append(source[j + 1:])
                    i = n
                continue
            i = j + 1
            continue
        if ch == "&":
            m = _ENTITY_REF_RE.match(source, i)
            if m:
                emit_entity(m.group(0))
                i = m.end()
            else:
                emit_entity("&")  # bare ampersand: exact literal
                i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out), entities, raw_tags


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
            if node.raw_start is not None:
                return node.raw_start
            return f"<{node.tag}{_serialize_attrs(node.attrs)}>"
        inner = "".join(_serialize_node(child) for child in node.children)
        if node.raw_start is not None and node.raw_end is not None:
            return f"{node.raw_start}{inner}{node.raw_end}"
        return f"<{node.tag}{_serialize_attrs(node.attrs)}>{inner}</{node.tag}>"
    return ""
