from __future__ import annotations

import posixpath
import shutil
import tempfile
import zipfile
from pathlib import Path

from lxml import etree as ET

from src.runtime.files import DEFAULT_BACKUP_DIR, digest

X = "http://www.w3.org/1999/xhtml"
N = "http://www.daisy.org/z3986/2005/ncx/"
O = "http://www.idpf.org/2007/opf"
NS = {"x": X, "n": N, "o": O}


def resolve(base: str, href: str) -> str:
    return posixpath.normpath(
        posixpath.join(posixpath.dirname(base), href.split("#")[0])
    )


def install_archive(
    candidate: Path, target: Path, expected_hash: str | None
) -> Path | None:
    """Replace only the observed file, retaining a backup of its exact bytes."""
    actual_hash = digest(target.read_bytes()) if target.exists() else None
    if actual_hash != expected_hash:
        raise ValueError(f"Local EPUB changed during export: {target}")
    backup = None
    if actual_hash:
        root = (Path.cwd() / "books").resolve()
        relative_path = (
            target.resolve().relative_to(root)
            if target.resolve().is_relative_to(root)
            else Path(target.name)
        )
        backup = DEFAULT_BACKUP_DIR / actual_hash[:12] / relative_path
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            shutil.copy2(target, backup)
        elif digest(backup.read_bytes()) != actual_hash:
            raise ValueError(f"EPUB backup does not match the original: {backup}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=target.parent, suffix=".epub", delete=False
    ) as file:
        temporary = Path(file.name)
    try:
        shutil.copy2(candidate, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return backup


def validate_archive(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if (
            not infos
            or infos[0].filename != "mimetype"
            or infos[0].compress_type != zipfile.ZIP_STORED
            or archive.read("mimetype") != b"application/epub+zip"
        ):
            raise ValueError("EPUB mimetype must be the first, uncompressed ZIP member")
        if archive.testzip() is not None:
            raise ValueError(f"Invalid ZIP: {path}")
        names = set(archive.namelist())
        if len(names) != len(infos):
            raise ValueError("Duplicate EPUB archive members")
        for name in names:
            if name.endswith((".xhtml", ".opf", ".ncx")):
                root = ET.fromstring(archive.read(name))
                if name.endswith(("nav.xhtml", ".ncx")):
                    for link in root.xpath("//*[@href or @src]"):
                        href = link.get("href") or link.get("src")
                        if ":" not in href and resolve(name, href) not in names:
                            raise ValueError(f"Broken EPUB navigation: {name}: {href}")
                if name.endswith(".opf"):
                    for item in root.find("o:manifest", NS):
                        href = item.get("href", "")
                        if not href or (
                            ":" not in href and resolve(name, href) not in names
                        ):
                            raise ValueError(f"Missing EPUB manifest resource: {href}")
                    ids = {node.get("id") for node in root.find("o:manifest", NS)}
                    spine = [node.get("idref") for node in root.find("o:spine", NS)]
                    if len(spine) != len(set(spine)) or not set(spine) <= ids:
                        raise ValueError("Invalid EPUB spine")


def validate_navigation(
    path: Path, nav_name: str, ncx_name: str, opf_name: str
) -> None:
    """Require the two complete TOC trees and the reading spine to agree."""
    with zipfile.ZipFile(path) as archive:
        nav = ET.fromstring(archive.read(nav_name))
        ncx = ET.fromstring(archive.read(ncx_name))
        opf = ET.fromstring(archive.read(opf_name))

    def nav_tree(nodes):
        return [
            (
                n.findtext("x:a", namespaces=NS),
                resolve(nav_name, n.find("x:a", NS).get("href")),
                nav_tree(n.findall("x:ol/x:li", NS)),
            )
            for n in nodes
        ]

    def ncx_tree(nodes):
        return [
            (
                n.findtext("n:navLabel/n:text", namespaces=NS),
                resolve(ncx_name, n.find("n:content", NS).get("src")),
                ncx_tree(n.findall("n:navPoint", NS)),
            )
            for n in nodes
        ]

    tree = nav_tree(nav.findall(".//x:nav/x:ol/x:li", NS))
    if tree != ncx_tree(ncx.findall("n:navMap/n:navPoint", NS)):
        raise ValueError("EPUB nav and NCX have different hierarchy, titles or order")

    def leaves(nodes):
        return [
            m
            for _, member, children in nodes
            for m in (leaves(children) if children else [member])
        ]

    expected = leaves(tree)
    manifest = {
        n.get("id"): resolve(opf_name, n.get("href"))
        for n in opf.find("o:manifest", NS)
    }
    actual = [manifest[n.get("idref")] for n in opf.find("o:spine", NS)]
    expected_set = set(expected)
    if (
        len(expected) != len(expected_set)
        or [m for m in actual if m in expected_set] != expected
    ):
        raise ValueError("EPUB spine differs from the complete chapter order")
