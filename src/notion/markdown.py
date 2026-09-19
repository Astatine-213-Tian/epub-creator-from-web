from __future__ import annotations

import re

from markdown_it import MarkdownIt

STYLES = ("bold", "italic", "underline", "strikethrough", "code")
INLINE = MarkdownIt("commonmark", {"html": True}).enable("strikethrough")


def inline_runs(text: str) -> list[dict]:
    runs: list[dict] = []
    styles: list[str] = []

    def append(value: str, extra: str | None = None) -> None:
        active = [style for style in STYLES if style in styles or style == extra]
        if runs and runs[-1]["styles"] == active:
            runs[-1]["text"] += value
        elif value:
            runs.append({"text": value, "styles": active})

    tokens = INLINE.parseInline(text)[0].children or []
    # Notion automatically links bare URLs. Keep those literal characters;
    # named links remain unsupported rather than discarding their destinations.
    autolink_delimiters = set()
    for index, token in enumerate(tokens[:-2]):
        if (
            token.type == "link_open"
            and tokens[index + 1].type == "text"
            and tokens[index + 1].content == token.attrGet("href")
            and tokens[index + 2].type == "link_close"
        ):
            autolink_delimiters.update((index, index + 2))
    for index, token in enumerate(tokens):
        if index in autolink_delimiters:
            continue
        if token.type == "text":
            append(token.content)
        elif token.type == "code_inline":
            append(token.content.replace("<br>", "\n"), "code")
        elif token.type in {"strong_open", "em_open", "s_open"}:
            styles.append(
                {"strong_open": "bold", "em_open": "italic", "s_open": "strikethrough"}[
                    token.type
                ]
            )
        elif token.type in {"strong_close", "em_close", "s_close"}:
            expected = {
                "strong_close": "bold",
                "em_close": "italic",
                "s_close": "strikethrough",
            }[token.type]
            if not styles or styles.pop() != expected:
                raise ValueError("Unbalanced Notion rich text")
        elif token.type in {"softbreak", "hardbreak"}:
            append("\n")
        elif token.type == "html_inline" and token.content in {
            "<br>",
            "<br/>",
            "<br />",
        }:
            append("\n")
        elif token.type == "html_inline" and token.content == '<span underline="true">':
            styles.append("underline")
        elif (
            token.type == "html_inline"
            and token.content == "</span>"
            and styles
            and styles[-1] == "underline"
        ):
            styles.pop()
        else:
            raise ValueError(
                f"Unsupported Notion rich text ({token.type}); EPUB was not changed"
            )
    if styles:
        raise ValueError("Unbalanced Notion rich text")
    return runs or [{"text": "", "styles": []}]


def escape(text: str) -> str:
    if re.fullmatch(r"(?:[-*_]\s*){3,}", text):
        return re.sub(r"([-*_])", r"\\\1", text)
    text = re.sub(r"([\\*_~`$\[\]<>{}|^])", r"\\\1", text)
    text = re.sub(r"^([#>+\-])(?=\s)", r"\\\1", text)
    return re.sub(r"^([0-9]+)([.)])(?=\s)", r"\1\\\2", text)


def to_markdown(blocks: list[dict]) -> str:
    lines = []
    wrappers = {
        "bold": ("**", "**"),
        "italic": ("*", "*"),
        "strikethrough": ("~~", "~~"),
        "underline": ('<span underline="true">', "</span>"),
        "code": ("`", "`"),
    }
    for block in blocks:
        if block["kind"] == "divider":
            line = "---"
        else:
            line = ""
            for run in block["runs"]:
                text = escape(run["text"]).replace("\n", "<br>")
                leading = text[: len(text) - len(text.lstrip())]
                trailing = text[len(text.rstrip()) :]
                if run["styles"]:
                    text = text.strip()
                for style in reversed(run["styles"]):
                    before, after = wrappers[style]
                    if text:
                        text = before + text + after
                if run["styles"]:
                    text = leading + text + (trailing if text else "")
                line += text
            if block["kind"] == "heading":
                line = "#" * block.get("level", 3) + " " + line
            elif block["kind"] == "quote":
                line = "> " + line
        line = line or "<empty-block/>"
        alignment = block.get("alignment")
        if alignment in {"center", "right"}:
            if alignment == "right":
                lines.extend(
                    [
                        "<columns>",
                        '\t<column ratio="25">',
                        "\t\t<empty-block/>",
                        "\t</column>",
                        '\t<column ratio="25">',
                        "\t\t<empty-block/>",
                        "\t</column>",
                        '\t<column ratio="50">',
                        "\t\t" + line,
                        "\t</column>",
                        "</columns>",
                    ]
                )
                continue
            lines.extend(
                [
                    "<columns>",
                    '\t<column ratio="25">',
                    "\t\t<empty-block/>",
                    "\t</column>",
                    '\t<column ratio="50">',
                    "\t\t" + line,
                    "\t</column>",
                    '\t<column ratio="25">',
                    "\t\t<empty-block/>",
                    "\t</column>",
                    "</columns>",
                ]
            )
        elif alignment not in (None, "left"):
            raise ValueError(
                f"Notion has no configured representation for {alignment} alignment"
            )
        else:
            lines.append(line)
    return "\n".join(lines)


def from_markdown(markdown: str) -> list[dict]:
    blocks = []
    lines = markdown.split("\n")
    index = 0
    while index < len(lines):
        line = lines[index]
        index += 1
        if not line:
            continue
        if line == "<columns>":
            columns = []
            while index < len(lines) and lines[index] != "</columns>":
                if not re.fullmatch(r'\t<column(?: ratio="[0-9.]+")?>', lines[index]):
                    raise ValueError("Unsupported column layout")
                index += 1
                content = []
                while index < len(lines) and lines[index].startswith("\t\t"):
                    content.append(lines[index][2:])
                    index += 1
                if index >= len(lines) or lines[index] != "\t</column>":
                    raise ValueError("Incomplete Notion column")
                index += 1
                columns.append(content)
            if index >= len(lines):
                raise ValueError("Incomplete Notion columns")
            index += 1
            occupied = [
                i
                for i, col in enumerate(columns)
                if any(l not in ("", "<empty-block/>") for l in col)
            ]
            if len(columns) != 3 or occupied not in ([1], [2]):
                raise ValueError(
                    "Alignment requires three columns with content in only the middle or right column"
                )
            middle = from_markdown("\n".join(columns[occupied[0]]))
            if not middle or not any(
                b["runs"] or b["kind"] == "divider" for b in middle
            ):
                raise ValueError("Centered column must contain content")
            blocks.extend(
                {**b, "alignment": "center" if occupied == [1] else "right"}
                for b in middle
            )
            continue
        if line == "<empty-block/>":
            blocks.append({"kind": "paragraph", "runs": []})
            continue
        if line == "---":
            blocks.append({"kind": "divider", "runs": []})
            continue
        heading = re.match(r"^(#{1,4}) (.+)$", line)
        block = {"kind": "paragraph"}
        if heading:
            line = heading[2]
            block.update(kind="heading", level=len(heading[1]))
        elif line.startswith("> "):
            line, block["kind"] = line[2:], "quote"
        elif re.match(r"^(?:\t|[-+*] |\d+[.)] |```|\$\$|!\[)", line):
            raise ValueError(
                "Unsupported chapter Markdown block; source was not changed"
            )
        if re.search(r"\{(?:color|toggle)=", line):
            raise ValueError("Unsupported Notion block formatting")
        block["runs"] = inline_runs(line)
        blocks.append(block)
    return blocks


def text_blocks(blocks: list[dict]) -> list[str]:
    return ["".join(run["text"] for run in b["runs"]) for b in blocks]
