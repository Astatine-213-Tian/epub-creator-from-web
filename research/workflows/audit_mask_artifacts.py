#!/usr/bin/env python3
from __future__ import annotations

"""Measure how much authorship accuracy comes from mask-token artifacts."""

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from .audit_style_dataset import CJK_RE
from .author_style_meter_contract import (
    CURRENT_BENCHMARK_RESULT,
    CURRENT_CHAR_MIN_DF,
    CURRENT_MAX_CHAR_FEATURES,
    RESEARCH_ROOT,
    file_sha256,
    normalize_char_text,
)
from .benchmark_author_style import (
    TARGET_AUTHOR_DEFAULT,
    chunk_paths,
    write_confusion_csv,
    write_svg_bar_chart,
)
from .benchmark_author_style_supervised import (
    CLASSIFIER_LABELS,
    ChunkRecord,
    MatrixBundle,
    RunSpec,
    evaluate_supervised_run,
    group_records_by_split,
    load_records,
    parse_csv_arg,
    safe_filename,
    validate_mask_plan,
    validate_view_alignment,
)


DEFAULT_OUTPUT_DIR = Path(
    "generated/style_research/benchmarks/mask_artifact_ablation_v2"
)
DEFAULT_PRIOR_RESULT = Path(
    "generated/style_research/benchmarks/mask_artifact_ablation_v1/"
    "mask_artifact_ablation_results.json"
)
MASK_RUN_RE = re.compile(r"某+")
MASK_RUN_BINS = (1, 2, 3, 4, 8, 16, 32)
GAP_BINS = (8, 16, 32, 64, 128, 256, 512)

REPRESENTATION_LABELS = {
    "collapsed_char_ngrams": "Exact 2-4 character n-grams after collapsing each mask run",
    "mask_stripped_char_ngrams": (
        "Exact 2-4 character n-grams within unmasked spans only"
    ),
    "mask_topology": "Mask density and run/gap topology only",
}

PROTOCOL = {
    "schema_version": 2,
    "purpose": "diagnose_mask_token_artifacts_and_lexical_topology_interactions",
    "primary_view": "train_global_masked",
    "primary_classifier": "sgd_hinge_unbalanced",
    "primary_representation": "mask_stripped_char_ngrams",
    "mask_stripped_stability": {
        "minimum_dev_accuracy": 0.80,
        "minimum_dev_balanced_accuracy": 0.80,
        "maximum_dev_accuracy_drop_from_original": 0.05,
        "minimum_test_balanced_accuracy": 0.80,
        "maximum_test_accuracy_drop_from_original": 0.05,
    },
    "selection_rule": (
        "Use development results and construct validity to assess the fixed global-only view. "
        "The prior collapsed-run and topology-only controls are frozen inputs. Test metrics "
        "diagnose stability only; the current test set is already consumed."
    ),
}


def collapse_mask_runs(text: str) -> str:
    """Canonicalize every length-preserving mask span to one marker."""
    return MASK_RUN_RE.sub("某", normalize_char_text(text))


def mask_stripped_char_ngrams(text: str) -> list[str]:
    """Yield lexical n-grams without mask markers or cross-mask bridge features."""
    features: list[str] = []
    for span in MASK_RUN_RE.split(normalize_char_text(text)):
        for size in range(2, 5):
            features.extend(
                span[index : index + size]
                for index in range(len(span) - size + 1)
            )
    return features


def bucket_label(value: int, boundaries: tuple[int, ...]) -> str:
    for boundary in boundaries:
        if value <= boundary:
            return f"le_{boundary}"
    return f"gt_{boundaries[-1]}"


def mask_topology_features(text: str) -> dict[str, float]:
    """Describe mask placement without retaining lexical or punctuation content."""
    cjk = "".join(CJK_RE.findall(text))
    total = len(cjk)
    if not total:
        return {"cjk_count": 0.0, "mask_fraction": 0.0, "run_count_per_1k": 0.0}

    runs = [match.end() - match.start() for match in MASK_RUN_RE.finditer(cjk)]
    gaps = [len(gap) for gap in MASK_RUN_RE.split(cjk) if gap]
    masked = sum(runs)
    features: dict[str, float] = {
        "cjk_count": float(total),
        "mask_fraction": masked / total,
        "run_count_per_1k": len(runs) * 1000.0 / total,
        "masked_cjk_per_1k": masked * 1000.0 / total,
    }
    if runs:
        features.update(
            {
                "run_length_mean": statistics.fmean(runs),
                "run_length_median": float(statistics.median(runs)),
                "run_length_max": float(max(runs)),
                "run_length_stdev": (
                    statistics.pstdev(runs) if len(runs) > 1 else 0.0
                ),
            }
        )
        run_counts = Counter(bucket_label(length, MASK_RUN_BINS) for length in runs)
        for label, count in run_counts.items():
            features[f"run_bucket:{label}_per_1k"] = count * 1000.0 / total
    if gaps:
        features.update(
            {
                "gap_length_mean": statistics.fmean(gaps),
                "gap_length_median": float(statistics.median(gaps)),
                "gap_length_max": float(max(gaps)),
            }
        )
        gap_counts = Counter(bucket_label(length, GAP_BINS) for length in gaps)
        for label, count in gap_counts.items():
            features[f"gap_bucket:{label}_per_1k"] = count * 1000.0 / total
    return features


def matrix_bundle(
    records: list[ChunkRecord],
    *,
    representation: str,
    max_char_features: int,
    char_min_df: int,
) -> MatrixBundle:
    records_by_split = group_records_by_split(records)
    train_records = records_by_split["train"]
    train_author_counts = Counter(record.author for record in train_records)

    if representation == "collapsed_char_ngrams":
        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 4),
            max_features=max_char_features,
            min_df=char_min_df,
            sublinear_tf=True,
            norm="l2",
            lowercase=False,
            preprocessor=collapse_mask_runs,
            dtype=np.float32,
        )
        train_matrix = vectorizer.fit_transform(record.text for record in train_records)
        matrices: dict[str, sparse.csr_matrix] = {"train": train_matrix.tocsr()}
        for split, split_records in records_by_split.items():
            if split != "train":
                matrices[split] = vectorizer.transform(
                    record.text for record in split_records
                ).tocsr()
        feature_count = len(vectorizer.get_feature_names_out())
    elif representation == "mask_stripped_char_ngrams":
        vectorizer = TfidfVectorizer(
            analyzer=mask_stripped_char_ngrams,
            max_features=max_char_features,
            min_df=char_min_df,
            sublinear_tf=True,
            norm="l2",
            lowercase=False,
            dtype=np.float32,
        )
        train_matrix = vectorizer.fit_transform(record.text for record in train_records)
        matrices = {"train": train_matrix.tocsr()}
        for split, split_records in records_by_split.items():
            if split != "train":
                matrices[split] = vectorizer.transform(
                    record.text for record in split_records
                ).tocsr()
        feature_count = len(vectorizer.get_feature_names_out())
    elif representation == "mask_topology":
        vectorizer = DictVectorizer(dtype=np.float32)
        scaler = StandardScaler(with_mean=False)
        train_counts = vectorizer.fit_transform(
            mask_topology_features(record.text) for record in train_records
        )
        train_matrix = scaler.fit_transform(train_counts).tocsr()
        matrices = {"train": train_matrix}
        for split, split_records in records_by_split.items():
            if split == "train":
                continue
            counts = vectorizer.transform(
                mask_topology_features(record.text) for record in split_records
            )
            matrices[split] = scaler.transform(counts).tocsr()
        feature_count = len(vectorizer.feature_names_)
    else:
        raise ValueError(f"unknown representation: {representation}")

    return MatrixBundle(
        labels=sorted(train_author_counts),
        train_author_counts=dict(train_author_counts),
        train_rows=len(train_records),
        feature_count=feature_count,
        records_by_split=records_by_split,
        matrix_by_split=matrices,
    )


def metric(result: dict[str, Any], split: str, name: str) -> float:
    value: Any = result["chunk_metrics"][split]
    for part in name.split("."):
        value = value[part]
    return float(value)


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def run_label(result: dict[str, Any]) -> str:
    spec = result["spec"]
    return (
        f"{spec['view']} / {REPRESENTATION_LABELS[spec['method']]} / "
        f"{CLASSIFIER_LABELS.get(spec['classifier'], spec['classifier'])}"
    )


def load_bound_evaluation(path: Path, result_path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    marker = f"<!-- ablation-result-sha256: {file_sha256(result_path)} -->"
    if marker not in text.splitlines()[:5]:
        raise SystemExit(f"{path} is not bound to the current ablation result")
    return text


def evaluate_protocol(payload: dict[str, Any]) -> dict[str, Any]:
    main = payload["main_benchmark_reference"]
    original = main["results"][
        "sgd_hinge_unbalanced.char_ngrams.train_global_masked"
    ]
    stripped = payload["results"][
        "sgd_hinge_unbalanced.mask_stripped_char_ngrams.train_global_masked"
    ]
    thresholds = PROTOCOL["mask_stripped_stability"]
    dev_accuracy_drop = (
        metric(original, "dev", "accuracy") - metric(stripped, "dev", "accuracy")
    )
    test_accuracy_drop = (
        metric(original, "test", "accuracy") - metric(stripped, "test", "accuracy")
    )
    checks = {
        "prior_controls_pass": bool(
            payload["prior_ablation_reference"]["protocol_evaluation"][
                "all_checks_pass"
            ]
        ),
        "stripped_dev_accuracy": metric(stripped, "dev", "accuracy")
        >= thresholds["minimum_dev_accuracy"],
        "stripped_dev_balanced_accuracy": metric(
            stripped, "dev", "balanced_accuracy"
        )
        >= thresholds["minimum_dev_balanced_accuracy"],
        "stripped_dev_accuracy_drop": dev_accuracy_drop
        <= thresholds["maximum_dev_accuracy_drop_from_original"],
        "stripped_test_balanced_accuracy": metric(
            stripped, "test", "balanced_accuracy"
        )
        >= thresholds["minimum_test_balanced_accuracy"],
        "stripped_test_accuracy_drop": test_accuracy_drop
        <= thresholds["maximum_test_accuracy_drop_from_original"],
    }
    return {
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "original_to_stripped_dev_accuracy_drop": dev_accuracy_drop,
        "original_to_stripped_test_accuracy_drop": test_accuracy_drop,
    }


def report_lines(
    payload: dict[str, Any],
    *,
    output_path: Path,
    chart_paths: dict[str, Path],
    evaluation: str,
) -> list[str]:
    results = payload["results"]
    rows = sorted(
        results.values(),
        key=lambda result: (
            result["spec"]["view"],
            result["spec"]["method"],
            result["spec"]["classifier"],
        ),
    )
    lines = [
        "# Mask-Artifact Ablation for the Authorship Proxy",
        "",
        "## Research Question",
        "",
        "Does the author classifier remain useful after every mask marker is removed and "
        "character n-grams are forbidden from crossing a removed span?",
        "",
        "## Protocol",
        "",
        "- Inputs: the fixed train-global mask and the global-plus-book-local mask.",
        "- Mask-stripped representation: normalize punctuation, split on every contiguous `某` "
        "  span, discard the markers, and extract exact 2-4 character TF-IDF features only "
        "  within the surviving spans.",
        "- Prior controls: the frozen v1 collapsed-run and topology-only result is hash-bound "
        "  into this run instead of being recomputed.",
        "- Models: class-balanced and unweighted SGD hinge, seed 13.",
        "- Split: the existing book-disjoint train/development/test/proxy assignment.",
        "- Selection: the test set is consumed. Development behavior and construct validity "
        "  determine the provisional view; test results are a stability diagnostic.",
        "",
        "## Results",
        "",
        "| View | Representation | Classifier | Dev acc. | Dev balanced | Test acc. | Test balanced | Target F1 | Book acc. |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in rows:
        spec = result["spec"]
        lines.append(
            f"| {spec['view']} | {REPRESENTATION_LABELS[spec['method']]} | "
            f"{CLASSIFIER_LABELS.get(spec['classifier'], spec['classifier'])} | "
            f"{percent(metric(result, 'dev', 'accuracy'))} | "
            f"{percent(metric(result, 'dev', 'balanced_accuracy'))} | "
            f"{percent(metric(result, 'test', 'accuracy'))} | "
            f"{percent(metric(result, 'test', 'balanced_accuracy'))} | "
            f"{percent(metric(result, 'test', 'target.f1'))} | "
            f"{percent(float(result['book_metrics']['test']['accuracy']))} |"
        )

    original = payload["main_benchmark_reference"]["results"][
        "sgd_hinge_unbalanced.char_ngrams.train_global_masked"
    ]
    stripped = results[
        "sgd_hinge_unbalanced.mask_stripped_char_ngrams.train_global_masked"
    ]
    protocol = payload["protocol_evaluation"]
    lines.extend(
        [
            "",
            f"![Ablation test accuracy]({chart_paths['accuracy'].relative_to(output_path.parent)})",
            "",
            f"![Ablation balanced accuracy]({chart_paths['balanced'].relative_to(output_path.parent)})",
            "",
            "## Registered Diagnostic Gate",
            "",
            f"- Original global-only unweighted test accuracy: {percent(metric(original, 'test', 'accuracy'))}",
            f"- Mask-stripped global-only unweighted dev accuracy: {percent(metric(stripped, 'dev', 'accuracy'))}",
            f"- Mask-stripped global-only unweighted test accuracy: {percent(metric(stripped, 'test', 'accuracy'))}",
            f"- Development accuracy drop: {protocol['original_to_stripped_dev_accuracy_drop'] * 100:+.2f}pp",
            f"- Test accuracy drop: {protocol['original_to_stripped_test_accuracy_drop'] * 100:+.2f}pp",
            f"- All preregistered diagnostic checks pass: **{str(protocol['all_checks_pass']).lower()}**",
            "",
        ]
    )
    for name, passed in protocol["checks"].items():
        lines.append(f"- `{name}`: {'pass' if passed else 'fail'}")
    lines.extend(
        [
            "",
            "Passing this diagnostic means the global-only attribution proxy does not require "
            "mask tokens, mask-run lengths, direct mask locations, or character n-grams that "
            "bridge masked spans. It does not validate the classifier as a continuous "
            "generated-text style meter.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "uv run author-style-research mask-artifact-audit --jobs -1",
            "uv run author-style-research mask-artifact-audit --report-only \\",
            "  --evaluation-file generated/style_research/benchmarks/mask_artifact_ablation_v2/evaluator_review.md",
            "```",
            "",
            "## Independent Evaluation",
            "",
            evaluation if evaluation else "_Pending a separately executed evaluator review._",
            "",
        ]
    )
    return lines


def write_outputs(
    payload: dict[str, Any],
    *,
    output_dir: Path,
    evaluation_file: Path | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "mask_artifact_ablation_results.json"
    result_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    chart_dir = output_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(payload["results"].values(), key=run_label)
    chart_paths = {
        "accuracy": chart_dir / "test_accuracy.svg",
        "balanced": chart_dir / "test_balanced_accuracy.svg",
    }
    write_svg_bar_chart(
        chart_paths["accuracy"],
        "Mask-Artifact Ablation: Test Accuracy",
        [(run_label(result), metric(result, "test", "accuracy")) for result in rows],
    )
    write_svg_bar_chart(
        chart_paths["balanced"],
        "Mask-Artifact Ablation: Test Balanced Accuracy",
        [
            (run_label(result), metric(result, "test", "balanced_accuracy"))
            for result in rows
        ],
    )

    evaluation = ""
    if evaluation_file and evaluation_file.exists():
        evaluation = load_bound_evaluation(evaluation_file, result_path)
    report_path = output_dir / "mask_artifact_ablation_results.md"
    report_path.write_text(
        "\n".join(
            report_lines(
                payload,
                output_path=report_path,
                chart_paths=chart_paths,
                evaluation=evaluation,
            )
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prior-result", type=Path, default=DEFAULT_PRIOR_RESULT)
    parser.add_argument(
        "--views", default="train_global_masked,entity_masked_v3"
    )
    parser.add_argument(
        "--representations", default="mask_stripped_char_ngrams"
    )
    parser.add_argument(
        "--classifiers", default="sgd_hinge,sgd_hinge_unbalanced"
    )
    parser.add_argument("--max-char-features", type=int, default=CURRENT_MAX_CHAR_FEATURES)
    parser.add_argument("--char-min-df", type=int, default=CURRENT_CHAR_MIN_DF)
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--random-state", type=int, default=13)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--evaluation-file", type=Path)
    args = parser.parse_args()

    result_path = args.output_dir / "mask_artifact_ablation_results.json"
    if args.report_only:
        if not result_path.is_file():
            raise SystemExit(f"missing ablation result: {result_path}")
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        write_outputs(
            payload, output_dir=args.output_dir, evaluation_file=args.evaluation_file
        )
        print(json.dumps({"report": str(result_path.with_suffix('.md'))}, indent=2))
        return 0

    if not CURRENT_BENCHMARK_RESULT.is_file():
        raise SystemExit(f"missing current benchmark: {CURRENT_BENCHMARK_RESULT}")
    if not args.prior_result.is_file():
        raise SystemExit(f"missing frozen prior ablation: {args.prior_result}")
    main_benchmark = json.loads(CURRENT_BENCHMARK_RESULT.read_text(encoding="utf-8"))
    prior_ablation = json.loads(args.prior_result.read_text(encoding="utf-8"))
    if not prior_ablation.get("protocol_evaluation", {}).get("all_checks_pass"):
        raise SystemExit("frozen prior ablation did not pass its registered controls")
    views = parse_csv_arg(args.views)
    representations = parse_csv_arg(args.representations)
    classifiers = parse_csv_arg(args.classifiers)
    unknown = set(representations) - set(REPRESENTATION_LABELS)
    if unknown:
        raise SystemExit(f"unknown representations: {', '.join(sorted(unknown))}")

    paths = chunk_paths(args.dataset_root)
    records_by_view = {view: load_records(paths[view]) for view in views}
    validate_view_alignment(records_by_view)
    mask_plan_path = args.dataset_root / "masked/mask_terms.json"
    validate_mask_plan(mask_plan_path, records_by_view[views[0]])

    input_bindings = {
        "main_benchmark": {
            "path": str(CURRENT_BENCHMARK_RESULT.relative_to(RESEARCH_ROOT)),
            "sha256": file_sha256(CURRENT_BENCHMARK_RESULT),
        },
        "mask_plan": {
            "path": str(mask_plan_path),
            "sha256": file_sha256(mask_plan_path),
        },
        "source": {
            "path": str(Path(__file__).resolve().relative_to(RESEARCH_ROOT)),
            "sha256": file_sha256(Path(__file__).resolve()),
        },
        "chunk_files": {
            view: {
                "path": str(paths[view]),
                "sha256": file_sha256(paths[view]),
                "rows": len(records_by_view[view]),
            }
            for view in views
        },
    }

    results: dict[str, dict[str, Any]] = {}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for view in views:
        for representation in representations:
            print(f"building {representation}.{view}", flush=True)
            bundle = matrix_bundle(
                records_by_view[view],
                representation=representation,
                max_char_features=args.max_char_features,
                char_min_df=args.char_min_df,
            )
            for classifier in classifiers:
                spec = RunSpec(
                    key=f"{classifier}.{representation}.{view}",
                    classifier=classifier,
                    method=representation,
                    view=view,
                )
                print(f"running {spec.key}", flush=True)
                result = evaluate_supervised_run(
                    bundle,
                    spec,
                    target_author=TARGET_AUTHOR_DEFAULT,
                    random_state=args.random_state,
                    max_iter=args.max_iter,
                    n_jobs=args.jobs,
                )
                results[spec.key] = result
                write_confusion_csv(
                    args.output_dir
                    / "confusion_matrices"
                    / f"{safe_filename(spec.key)}.test.csv",
                    result["labels"],
                    result["test_confusion_matrix"],
                )
            del bundle

    payload: dict[str, Any] = {
        "protocol": PROTOCOL,
        "target_author": TARGET_AUTHOR_DEFAULT,
        "views": views,
        "representations": representations,
        "classifiers": classifiers,
        "char_min_df": args.char_min_df,
        "max_char_features": args.max_char_features,
        "input_bindings": input_bindings,
        "main_benchmark_reference": {
            "sha256": file_sha256(CURRENT_BENCHMARK_RESULT),
            "results": main_benchmark["results"],
        },
        "prior_ablation_reference": {
            "path": str(args.prior_result),
            "sha256": file_sha256(args.prior_result),
            "protocol_evaluation": prior_ablation["protocol_evaluation"],
        },
        "results": results,
    }
    payload["protocol_evaluation"] = evaluate_protocol(payload)
    write_outputs(payload, output_dir=args.output_dir, evaluation_file=None)
    print(
        json.dumps(
            {
                "results": str(result_path),
                "report": str(result_path.with_suffix('.md')),
                "runs": len(results),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
