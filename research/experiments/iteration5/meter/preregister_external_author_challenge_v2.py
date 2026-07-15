#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path


OUTPUT = Path(
    "generated/style_research/style_transfer_experiments/iterations/"
    "content_resistant_v1/cr_fysm_v4/external_confirmation_v2/"
    "preregistration.external.v2.json"
)
DATASET_ROOT = Path("datasets/external_author_challenge_v2")
SEED = 20260716
ROSTER = [
    {
        "author": "易人北",
        "official_author_page": "https://m.jjwxc.net/wapauthor/1075777",
        "verified_chunai_titles": ["异世流放", "跳梁小丑混世记", "丑皇"],
        "selected": [
            ["异世流放", "https://quanben.io/n/yishiliufang/", "架空历史", "奇幻"],
            ["跳梁小丑混世记", "https://quanben.io/n/tiaoliangxiaochouhunshiji/", "架空历史", "仙侠"],
        ],
    },
    {
        "author": "来自远方",
        "official_author_page": "https://m.jjwxc.net/wapauthor/18719",
        "verified_chunai_titles": ["异世大领主", "星际童话", "重生成猎豹"],
        "selected": [
            ["异世大领主", "https://quanben.io/n/yishidalingzhu/", "架空历史", "爱情"],
            ["星际童话", "https://quanben.io/n/xingjitonghua/", "近代现代", "科幻"],
        ],
    },
    {
        "author": "酥油饼",
        "official_author_page": "https://m.jjwxc.net/wapauthor/348952",
        "verified_chunai_titles": ["亡迹", "识汝不识丁", "有珠何须椟"],
        "selected": [
            ["亡迹", "https://quanben.io/n/wangji/", "架空历史", "奇幻"],
            ["识汝不识丁", "https://quanben.io/n/shirubushiding/", "古色古香", "爱情"],
        ],
    },
    {
        "author": "耳雅",
        "official_author_page": "https://m.jjwxc.net/wapauthor/281027",
        "verified_chunai_titles": ["SCI谜案集", "好木望天", "黄半仙=活神仙"],
        "selected": [
            ["SCI谜案集", "https://quanben.io/n/scimianji/", "近代现代", "古典衍生"],
            ["好木望天", "https://quanben.io/n/haomuwangtian/", "古色古香", "爱情"],
        ],
    },
    {
        "author": "决绝",
        "official_author_page": "https://m.jjwxc.net/wapauthor/279209",
        "verified_chunai_titles": ["末世之功德无量", "重生之异兽猎人", "重临巅峰"],
        "selected": [
            ["末世之功德无量", "https://quanben.io/n/moshizhigongdewuliang/", "幻想未来", "爱情"],
            ["重生之异兽猎人", "https://quanben.io/n/chongshengzhiyishoulieren/", "幻想未来", "科幻"],
        ],
    },
]


def canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def main() -> None:
    payload = {
        "schema_version": 1,
        "challenge_id": "CR-FYSM-v4-external-author-confirmation-v2",
        "status": "locked_before_download_cleanup_masking_and_scoring",
        "seed": SEED,
        "dataset_root": str(DATASET_ROOT),
        "collection_policy": {
            "output_mode": "txt_only",
            "epub_output_forbidden": True,
            "books_directory_forbidden": True,
            "overwrite_forbidden": True,
            "authors": 5,
            "books_per_author": 2,
            "eligibility": "at least three official JJWXC 纯爱 books",
            "source_validation": "exact title/author and completed Quanben TOC preview",
        },
        "roster": ROSTER,
        "exclusions": {
            "active_training_authors": "all authors in datasets/dataset_manifest.json",
            "cr_fysm_v3_external_authors": [
                "MRA",
                "南康白起",
                "壶鱼辣椒",
                "孔恰",
                "忆若甫",
                "思皓凡",
                "戏子祭酒",
                "水千丞",
                "燕孤鸿",
                "狮子星系",
            ],
            "minimum_title_fingerprint_overlap": 0,
            "minimum_clean_text_fingerprint_overlap": 0,
        },
        "preprocessing": {
            "cleanup_script": "workflows/audit_style_dataset.py",
            "masking_view": "entity_masked_v3",
            "boundary_chunks_excluded": 2,
            "minimum_cleaned_cjk_per_book": 50000,
            "minimum_retained_books_per_author": 2,
        },
        "frozen_meter": {
            "meter_id": "CR-FYSM-v4",
            "threshold": 0.49830370600012425,
            "refit": False,
            "threshold_retuning": False,
        },
        "gates": {
            "overall_author_book_weighted_specificity_min": 0.90,
            "each_author_specificity_min": 0.90,
            "each_book_specificity_min": 0.80,
            "author_cluster_bootstrap_lower_min": 0.90,
            "clean_masked_threshold_flip_max": 0.05,
            "no_missing_or_excluded_authors": True,
        },
        "interpretation": (
            "A pass confirms external specificity only. Target-style sensitivity and "
            "generated-domain construct validity are established by separate gates."
        ),
    }
    payload["lock_id"] = hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "lock_id": payload["lock_id"]}, indent=2))


if __name__ == "__main__":
    main()
