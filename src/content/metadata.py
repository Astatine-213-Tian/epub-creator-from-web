"""Resolve official book metadata before creating an editable source."""

from __future__ import annotations

from src.metadata.catalog import MetadataLookup, PackageMetadata


def enrich_source(book: dict, *, lookup: MetadataLookup | None = None) -> dict:
    data = book["metadata"]
    package = PackageMetadata(
        title=data["title"],
        author=data["creator"],
        language=data["language"],
        date=data["date"],
        source=data["source"],
        description=data["description"],
        subjects=tuple(data["subjects"]),
        series=data["series"] or None,
        series_position=str(data["series_position"])
        if data["series_position"]
        else None,
    )
    source = (lookup or MetadataLookup()).find(package, primary_subject="")
    if source is None:
        return {"status": "unmatched"}
    data.update(
        title=source.title,
        language=source.language,
        date=source.date,
        source=source.source,
        description=source.description or data["description"],
        subjects=list(source.subjects),
    )
    if source.series_verified:
        data.update(
            series=source.series or "", series_position=source.series_position or ""
        )
    return {"status": "matched", "provider": source.provider, "url": source.source}
