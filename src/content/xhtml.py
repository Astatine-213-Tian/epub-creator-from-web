from __future__ import annotations

import re

from lxml import etree as ET

X = "http://www.w3.org/1999/xhtml"
INLINE_TAGS = {
    "strong": "bold",
    "b": "bold",
    "em": "italic",
    "i": "italic",
    "u": "underline",
    "s": "strikethrough",
    "code": "code",
}


def xml_runs(element: ET._Element) -> list[dict]:
    runs: list[dict] = []

    def add(text: str | None, styles: list[str]) -> None:
        if not text:
            return
        if runs and runs[-1]["styles"] == styles:
            runs[-1]["text"] += text
        else:
            runs.append({"text": text, "styles": styles})

    def visit(node: ET._Element, styles: list[str]) -> None:
        add(node.text, styles)
        for child in node:
            tag = ET.QName(child).localname
            if tag == "br":
                add("\n", styles)
            elif tag in INLINE_TAGS:
                visit(child, list(dict.fromkeys([*styles, INLINE_TAGS[tag]])))
            elif tag == "span" and not child.attrib:
                visit(child, styles)
            else:
                raise ValueError(f"Unsupported inline XHTML: {tag}")
            add(child.tail, styles)

    visit(element, [])
    return runs


def chapter_container(data: bytes) -> tuple[ET._Element, ET._Element]:
    root = ET.fromstring(data)
    body = root.find(f"{{{X}}}body")
    if body is None:
        raise ValueError("Missing chapter body")
    container = body
    if len(body) == 1 and ET.QName(body[0]).localname == "div":
        container = body[0]
    return body, container


def read_xhtml(data: bytes, *, title: str) -> list[dict]:
    _, container = chapter_container(data)
    nodes = list(container)
    if not nodes or ET.QName(nodes[0]).localname not in {"h1", "h2", "h3"}:
        raise ValueError("Missing chapter heading")
    if "".join(nodes.pop(0).itertext()).strip() != title:
        raise ValueError(f"Chapter title disagrees with EPUB heading: {title}")
    result = []
    for node in nodes:
        tag = ET.QName(node).localname
        if tag not in {"p", "h1", "h2", "h3", "h4", "blockquote", "hr"}:
            raise ValueError(f"Unsupported chapter block: {tag}")
        kind = (
            "heading"
            if tag.startswith("h") and tag != "hr"
            else {"p": "paragraph", "blockquote": "quote", "hr": "divider"}.get(tag)
        )
        block = {"kind": kind, "runs": xml_runs(node)}
        if kind == "heading":
            block["level"] = int(tag[1])
        attrs = dict(node.attrib)
        style = attrs.get("style", "")
        if tag == "h3":
            # The shared H3 rule owns size; it is not source-specific metadata.
            style = re.sub(
                r"(?:^|;)\s*font-size\s*:[^;]*(?:;|$)", ";", style, flags=re.I
            ).strip("; ")
            if style:
                attrs["style"] = style + ";"
            else:
                attrs.pop("style", None)
        alignment = re.search(
            r"(?:^|;)\s*text-align\s*:\s*(left|right|center|justify)\b", style
        )
        if alignment:
            block["alignment"] = alignment[1]
            # Alignment is content metadata, not tied to a paragraph's words.
            style = re.sub(r"(?:^|;)\s*text-align\s*:[^;]*(?:;|$)", ";", style)
            if alignment[1] == "center":
                style = re.sub(
                    r"(?:^|;)\s*text-indent\s*:\s*0(?:px|em)?\s*(?:;|$)", ";", style
                )
            if style.strip("; "):
                attrs["style"] = style.strip("; ") + ";"
            else:
                attrs.pop("style", None)
        if attrs:
            block["attributes"] = attrs
        result.append(block)
    return result
