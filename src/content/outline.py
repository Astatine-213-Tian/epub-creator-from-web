from __future__ import annotations


def ordered_members(sections: list[dict], depth: int = 1) -> list[str]:
    if depth > 3:
        raise ValueError("Book outline supports at most three levels")
    result = []
    for node in sections:
        if "children" in node:
            if not node["children"]:
                raise ValueError("Outline contains an empty heading")
            result.extend(ordered_members(node["children"], depth + 1))
        else:
            result.append(node["member"])
    return result
