"""Prepare source formatting once, before it becomes editable Notion content."""

from __future__ import annotations

import copy
import posixpath

import cssselect2
import tinycss2
from lxml import etree as ET

from src.content.xhtml import X

NS = {"x": X}


def declarations(tokens) -> dict[str, str]:
    return {
        item.lower_name: tinycss2.serialize(item.value).strip()
        for item in tinycss2.parse_declaration_list(
            tokens, skip_comments=True, skip_whitespace=True
        )
        if item.type == "declaration"
    }


def prepare_blocks(
    data: bytes, member: str, files: dict[str, bytes], blocks: list[dict]
) -> list[dict]:
    """Resolve source CSS alignment and promote existing subheadings to H3.

    Arbitrary source classes and inline sizes do not travel into the renderer.
    Notion stores the resulting heading, emphasis and alignment explicitly.
    """
    root = ET.fromstring(data)
    styles = [n.text or "" for n in root.findall(".//x:style", NS)]
    for link in root.findall(".//x:link[@rel='stylesheet']", NS):
        styles.append(
            files[
                posixpath.normpath(
                    posixpath.join(
                        posixpath.dirname(member), link.get("href").split("#")[0]
                    )
                )
            ].decode("utf-8")
        )
    matcher = cssselect2.Matcher()
    for rule in tinycss2.parse_stylesheet(
        "\n".join(styles), skip_comments=True, skip_whitespace=True
    ):
        if rule.type != "qualified-rule":
            continue
        for selector in cssselect2.compile_selector_list(rule.prelude):
            matcher.add_selector(selector, declarations(rule.content))
    computed = {}
    for element in cssselect2.ElementWrapper.from_xml_root(root).iter_subtree():
        values = {}
        for _, _, pseudo, payload in matcher.match(element):
            if pseudo is None:
                values.update(payload)
        values.update(declarations(element.etree_element.get("style", "")))
        computed[element.etree_element] = values
    container = root.find("x:body", NS)
    if len(container) == 1 and ET.QName(container[0]).localname == "div":
        container = container[0]
    result = copy.deepcopy(blocks)
    for node, block in zip(list(container)[1:], result, strict=True):
        for run in block["runs"]:
            # Notion drops Unicode line/paragraph separators. Store the same
            # visual break explicitly so the Markdown adapter emits <br>.
            run["text"] = run["text"].replace("\u2028", "\n").replace("\u2029", "\n")
        # Paragraph indentation is presentation. Notion interprets leading
        # ASCII whitespace as block nesting, so remove it during preparation.
        while block["runs"]:
            block["runs"][0]["text"] = block["runs"][0]["text"].lstrip(" \t")
            if block["runs"][0]["text"]:
                break
            block["runs"].pop(0)
        style = computed[node]
        alignment = style.get("text-align")
        if alignment in {"left", "center", "right"}:
            block["alignment"] = alignment
        if block["kind"] == "heading":
            block["level"] = 3
        elif (
            block["kind"] == "paragraph"
            and block.get("alignment") == "center"
            and block["runs"]
            and all("bold" in r["styles"] for r in block["runs"])
        ):
            block.update(kind="heading", level=3)
        block.pop("attributes", None)
    return result
