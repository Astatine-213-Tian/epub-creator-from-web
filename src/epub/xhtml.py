from __future__ import annotations

import re

from lxml import etree as ET

X = "http://www.w3.org/1999/xhtml"


def render_blocks(parent: ET._Element, blocks: list[dict]) -> None:
    for block in blocks:
        tag = {
            "paragraph": "p",
            "heading": "h" + str(block.get("level", 3)),
            "quote": "blockquote",
            "divider": "hr",
        }[block["kind"]]
        el = ET.SubElement(parent, f"{{{X}}}{tag}", **block.get("attributes", {}))
        # Content H3 has one global size, independent of alignment and template.
        if tag == "h3":
            style = re.sub(
                r"(?:^|;)\s*font-size\s*:[^;]*(?:;|$)",
                ";",
                el.get("style", ""),
                flags=re.I,
            ).strip("; ")
            el.set("style", (style + "; " if style else "") + "font-size: 1.1em;")
        alignment = block.get("alignment", "left" if tag == "h3" else None)
        if alignment:
            if alignment not in {"left", "center", "right", "justify"}:
                raise ValueError("Unsupported paragraph alignment")
            style = el.get("style", "").rstrip("; ")
            if style:
                style += "; "
            style += f"text-align: {alignment};"
            if alignment in {"center", "right"}:
                style += " text-indent: 0;"
            el.set("style", style)
        for run in block["runs"]:
            target = el
            for style in run["styles"]:
                name = {
                    "bold": "strong",
                    "italic": "em",
                    "underline": "u",
                    "strikethrough": "s",
                    "code": "code",
                }[style]
                target = ET.SubElement(target, f"{{{X}}}{name}")
            for index, text in enumerate(run["text"].split("\n")):
                if index:
                    ET.SubElement(target, f"{{{X}}}br")
                if len(target):
                    target[-1].tail = (target[-1].tail or "") + text
                else:
                    target.text = (target.text or "") + text
