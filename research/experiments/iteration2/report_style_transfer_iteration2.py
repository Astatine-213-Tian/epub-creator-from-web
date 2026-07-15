from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


from experiments.shared.paths import RESEARCH_ROOT


PROJECT_ROOT = RESEARCH_ROOT
DEFAULT_ROOT = Path(
    "generated/style_research/style_transfer_experiments/iterations/aligned_pairs_v1"
)
DEFAULT_OUTPUT = Path(
    "docs/reports/04_transfer_iteration2_aligned_pairs.md"
)
METHOD_ORDER = [
    "neutral_only",
    "aligned_pairs_only",
    "aligned_pairs_plus_cards",
    "aligned_pairs_edit_plan",
]
ARM_LABELS = {
    "own_author_reconstruction": "Own-author reconstruction",
    "cross_author_transfer": "Cross-author transfer",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Eternal Gate iteration-two style-transfer report."
    )
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def absolute(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def fmt_number(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.{digits}f}"


def fmt_percent(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.{digits}f}%"


def qa_progression(run_root: Path) -> list[tuple[str, int, int]]:
    rows: list[tuple[str, int, int]] = []
    initial = sorted((run_root / "english_source_qa").glob("*.json"))
    if initial:
        unresolved = sum(
            not read_json(path).get("approved_for_neutral_translation", False)
            for path in initial
        )
        rows.append(("Initial QA", len(initial), unresolved))
    repair_root = run_root / "english_source_repair_qa"
    for directory in sorted(repair_root.glob("round_*")):
        files = sorted(directory.glob("*.json"))
        if not files:
            continue
        unresolved = sum(
            not read_json(path).get("approved_for_neutral_translation", False)
            for path in files
        )
        round_number = int(directory.name.removeprefix("round_"))
        rows.append((f"Repair QA round {round_number}", len(files), unresolved))
    return rows


def ledger_stats(path: Path) -> dict[str, int]:
    rows = read_jsonl(path)
    failures = [row for row in rows if row.get("status") == "failed"]
    return {
        "requests": len(rows),
        "successes": sum(row.get("status") == "success" for row in rows),
        "failed_attempts": len(failures),
        "failed_samples": len({row["sample_id"] for row in failures}),
        "retried_samples": len(
            {row["sample_id"] for row in rows if int(row.get("attempt", 1)) > 1}
        ),
        "timeouts": sum(
            any("timed out" in error for error in row.get("validation_errors", []))
            for row in failures
        ),
    }


def method_results(root: Path) -> dict[str, dict[str, Any]]:
    evaluation = root / "evaluations/screening_v1_initial/methods"
    results: dict[str, dict[str, Any]] = {}
    for method_id in METHOD_ORDER:
        intensity = "none" if method_id == "neutral_only" else "medium"
        results[method_id] = read_json(
            evaluation / method_id / intensity / "results.json"
        )
    return results


def fidelity_reasons(result: dict[str, Any]) -> Counter[str]:
    reasons: Counter[str] = Counter()
    for row in result["rows"]:
        for paragraph in row.get("fidelity_gate", {}).get("paragraphs", []):
            reasons.update(paragraph.get("failures", []))
    return reasons


def audit_summary(path: Path) -> tuple[str, str]:
    if not path.exists():
        return "pending", "The separate outcome audit had not finished when this report ran."
    text = path.read_text(encoding="utf-8")
    verdict_match = re.search(
        r"(?im)^(?:verdict|decision)\s*:\s*\*\*(.+?)\*\*", text
    )
    if verdict_match:
        verdict = verdict_match.group(1).strip()
    else:
        verdict_section = re.search(
            r"(?ims)^## Verdict\s+(.+?)(?:\n## |\Z)", text
        )
        verdict = (
            verdict_section.group(1).strip().splitlines()[0].rstrip(".")
            if verdict_section
            else "complete"
        )
    close_read_match = re.search(
        r"(?i)(?:close[- ]read(?:ing)?|reviewed)\D{0,40}(\d+)\s+"
        r"(?:samples|outputs|cases|method-output cells|cells)",
        text,
    )
    detail = "Independent English-grounded close reading completed."
    if close_read_match:
        detail = (
            "Independent English-grounded close reading covered "
            f"{close_read_match.group(1)} method-output cells."
        )
    return verdict, detail


def build_report(root: Path, output: Path) -> None:
    evaluation_root = root / "evaluations/screening_v1_initial"
    evaluation = read_json(evaluation_root / "evaluation_summary.json")
    pair_summary = read_json(
        root / "sample_sets/iteration2_aligned_pair_pool_v1.summary.json"
    )
    proxy_summary = read_json(root / "sample_sets/development_proxy_v1.summary.json")
    preregistration = read_json(root / "protocols/iteration2_preregistration.v1.json")
    model_config = read_json(root / "protocols/model_run_config.v1.json")
    rerank = read_json(
        evaluation_root / "diagnostics/family_rerank/family_rerank_summary.json"
    )
    methods = method_results(root)
    method_configs = {
        method_id: read_json(root / f"method_registry/methods/{method_id}.v1.json")
        for method_id in METHOD_ORDER
    }

    pair_run = root / "runs/iteration2_aligned_pair_pool_v1/aligned_pair_pool_gpt54_v1"
    screen_run = root / "runs/development_proxy_v1/iteration2_aligned_gpt55_v1"
    active_lock = (
        "protocols/pre_style_analysis_lock.v1."
        "e97a085379b0cefe15bd60bd7d9d539c73f23b283e8c2d8c19aa805e209a7d4f.json"
    )
    audit_path = root / "audits/iteration2_outcome_audit.md"
    audit_verdict, audit_detail = audit_summary(audit_path)

    retry_stats = {
        method_id: ledger_stats(
            screen_run
            / "ledgers"
            / f"style_transfer.{method_id}.medium.jsonl"
        )
        for method_id in METHOD_ORDER
        if method_id != "neutral_only"
    }
    pair_qa = qa_progression(pair_run)
    screen_qa = qa_progression(screen_run)
    threshold = evaluation["threshold"]
    old_threshold = threshold.get("artifact", {}).get("selected", {}).get("threshold")
    scene_distribution = ", ".join(
        f"{name.replace('_', ' ')}: {count}"
        for name, count in pair_summary["scene_counts"].items()
    )

    chart_root = (
        "../../generated/style_research/style_transfer_experiments/iterations/"
        "aligned_pairs_v1/evaluations/screening_v1_initial"
    )
    now = datetime.now(UTC).isoformat(timespec="seconds")
    lines: list[str] = [
        "# Eternal Gate Style-Transfer Research: Iteration 2",
        "",
        f"Generated from frozen artifacts at `{now}`.",
        "",
        "## Executive Result",
        "",
        "Iteration 2 finished source construction, neutral translation, three aligned-pair "
        "style-transfer arms, deterministic scoring, and an independent outcome audit. All three "
        "style methods produced positive mean target-author margin lift in both benchmark arms, "
        "but **none passed the preregistered joint gate** because own-author hard-fidelity failure "
        "rates were 25.0%-29.2%, above the 10% limit. Confirmation was therefore not opened.",
        "",
        "The reused binary style-success threshold is also invalid for this iteration because its "
        "allocation and evaluation-protocol hashes still bind iteration 1. The report therefore "
        "does not claim a threshold-based accuracy or an 80% result. This threshold defect cannot "
        "be repaired post hoc after viewing outputs; it must be corrected before iteration 3.",
        "",
        "| Stage | Data | Result | Status |",
        "| --- | --- | --- | --- |",
        f"| Aligned-pair evidence | {pair_summary['total_samples']} excerpts from {pair_summary['books']} target-author train books | {pair_summary['total_samples']} English -> neutral -> original-target pairs | complete |",
        f"| Fresh screen | {preregistration['proxy_screen']['screening_own']} own-author + {preregistration['proxy_screen']['screening_rows'] - preregistration['proxy_screen']['screening_own']} cross-author rows | Zero overlap with iteration-1 screen | complete |",
        "| Style generation | 3 methods x 36 rows | 108/108 final outputs valid | complete |",
        "| Deterministic evaluation | Frozen exact character n-gram style meter plus fidelity/copy gates | Positive relative style lift; no copy failures; every method fails joint promotion | complete |",
        f"| Independent evaluator | Separate `gpt-5.5` close reading and audit | {audit_verdict}; {audit_detail} | {'complete' if audit_path.exists() else 'pending'} |",
        f"| Confirmation | {preregistration['proxy_screen']['confirmation_rows']} untouched rows | Not generated because no base method qualified | locked |",
        "",
        "## Research Question",
        "",
        "Do retrieved aligned neutral-to-target demonstrations teach the target author's "
        "transformations more effectively than iteration-1 unaligned examples while preserving "
        "the English source meaning and Chinese surface structure?",
        "",
        "## Experimental Design",
        "",
        "### Data roles",
        "",
        "| Role | Construction | Exposure |",
        "| --- | --- | --- |",
        f"| In-context pair pool | One masked excerpt from each of {pair_summary['books']} 非天夜翔 train books; scenes: {scene_distribution} | Available only as retrieved style evidence; no development or test books |",
        "| Pair English source | `gpt-5.4` reconstruction from each masked Chinese train excerpt | Used to create production-direction neutral Chinese |",
        "| Pair neutral Chinese | `gpt-5.4` using the same `eternal_gate_pass_1` neutral prompt contract | Retrieval query and neutral side of each aligned demonstration |",
        "| Pair target Chinese | Original `entity_masked_v2` train excerpt | Target side of an in-context pair; never an evaluation target |",
        f"| Iteration-2 screen | {proxy_summary['iteration2_screening_samples']} fresh rows: 3 per target-author proxy book and 1 per comparison author | Used once for method screening |",
        f"| Confirmation | {proxy_summary['iteration2_confirmation_samples']} rows: 80 own-author and 36 cross-author | Unopened by style generators |",
        "| Final validation | Four target-author books reserved in the corpus protocol | Unopened; no row IDs created |",
        "",
        "This is **in-context style transfer, not model fine-tuning**. The phrase `training pairs` "
        "means prompt evidence retrieved from train-split books. Model weights were not updated.",
        "",
        "### Why English and neutral Chinese were generated",
        "",
        "Production starts from the English Eternal Gate translation. A Chinese-to-Chinese "
        "neutralization benchmark would give the transfer method an easier and different input "
        "distribution. Iteration 2 therefore reconstructs English semantic sources and applies the "
        "same frozen neutral-translation prompt intended for production before any style rewrite.",
        "",
        "### Source QA and repair",
        "",
        "A repair round regenerated only English sources still rejected by semantic QA. It did not "
        "edit style outputs, did not repeatedly change already approved rows, and did not use style "
        "scores. The pair pool required three repair rounds; the fresh screen required six.",
        "",
        "| Dataset | QA point | Rows checked | Still unresolved |",
        "| --- | --- | ---: | ---: |",
    ]
    for dataset, progression in (("Aligned-pair pool", pair_qa), ("Fresh screen", screen_qa)):
        for label, checked, unresolved in progression:
            lines.append(f"| {dataset} | {label} | {checked} | {unresolved} |")

    model_selection = model_config["model_selection"]
    lines.extend(
        [
            "",
            "### Models and frozen execution",
            "",
            "| Component | Model | Reasoning | Notes |",
            "| --- | --- | --- | --- |",
            f"| English reconstruction, QA, repair, neutral Chinese | `{model_selection['neutral_reconstruction_model']}` | high | Same neutral prompt contract as planned production pass 1 |",
            f"| All style-transfer arms | `{model_selection['style_transfer_model']}` | high | One model across arms; no model mixing |",
            "| Independent methodology and outcome reviews | `gpt-5.5` | high | Separate Codex sessions; no style generation |",
            "",
            f"`gpt-5.6` was tried first and returned `{model_selection['gpt-5.6_probe']}`. "
            f"`gpt-5.5` returned `{model_selection['gpt-5.5_probe']}` and was used for style transfer.",
            "",
            f"The active pre-generation analysis lock is `{active_lock}`. A separate methodology "
            "reviewer issued **GO** only after stale source bindings, sample hashes, evaluator "
            "geometry, and promotion wording were corrected.",
            "",
            "### What the pair rebuild changed",
            "",
            "The audit-triggered `prepare.py` rebuild did **not** call an LLM and did not replace "
            f"the {pair_summary['total_samples']} English, neutral, or target texts. It deterministically rewrote manifests, role "
            "labels, paths, hashes, method assets, source bindings, and the active lock so every "
            "artifact referenced the iteration-2 allocation. Rebuilding those records was required "
            "because the first frozen package still contained iteration-1 paths and geometry.",
            "",
            "## Methods Tested",
            "",
            "| Method | Meaning | Evidence visible to `gpt-5.5` | Hypothesis |",
            "| --- | --- | --- | --- |",
        ]
    )
    method_meanings = {
        "neutral_only": "No rewrite; score the neutral Chinese unchanged as the paired control.",
        "aligned_pairs_only": "Retrieve three semantically similar neutral-to-target pairs, with at most one pair from each source book.",
        "aligned_pairs_plus_cards": "Use the same three aligned pairs plus compact corpus style cards only when cards agree with pair evidence.",
        "aligned_pairs_edit_plan": "Internally infer recurring edits in clause order, sentence boundaries, dialogue timing, function words, and reaction beats; apply only patterns supported by at least two pairs.",
    }
    for method_id in METHOD_ORDER:
        config = method_configs[method_id]
        inputs = ", ".join(f"`{value}`" for value in config["inputs"])
        lines.append(
            f"| `{method_id}` | {method_meanings[method_id]} | {inputs} | {config['hypothesis']} |"
        )

    lines.extend(
        [
            "",
            "Every method received byte-identical English and neutral Chinese. The style runner "
            "could not read original evaluation Chinese, author/book labels, target-derived strata, "
            "or scores. Retrieval used only target-author train books and capped evidence at one pair "
            "per book.",
            "",
            "## Generation Reliability",
            "",
            "All arms ultimately produced 36 schema-valid outputs. Long samples with 46-53 "
            "paragraphs caused paragraph-ID omissions, duplicates, and occasional timeouts, so the "
            "request count is materially larger than the final output count.",
            "",
            "| Method | Final outputs | Total requests | Failed attempts | Samples needing retry | Timeouts |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for method_id in METHOD_ORDER[1:]:
        stats = retry_stats[method_id]
        lines.append(
            f"| `{method_id}` | {stats['successes']}/36 | {stats['requests']} | "
            f"{stats['failed_attempts']} | {stats['retried_samples']} | {stats['timeouts']} |"
        )

    lines.extend(
        [
            "",
            "This is an execution-quality result, not a style result. Iteration 3 should freeze a "
            "paragraph-batched long-form execution method before generation so output-schema "
            "retries do not depend on repeated whole-chunk calls.",
            "",
            "## Deterministic Results",
            "",
            "The frozen Stage-1 style meter is the class-balanced exact 2-4 character n-gram SGD "
            "hinge classifier on `entity_masked_v3`. `Mean paired lift` is candidate target-author "
            "decision margin minus the same row's neutral margin. Positive lift means movement toward "
            "the target-author side of this proxy; it is not a probability or human style score.",
            "",
            "![Mean paired margin lift](" + chart_root + "/charts/mean_paired_lift.svg)",
            "",
            "| Method | Arm | Mean target margin | Mean paired lift | Cluster bootstrap 95% CI | Positive rows | Target top-class share | Fidelity failures | Copy failures |",
            "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for method_id in METHOD_ORDER:
        result = methods[method_id]
        for arm in ("own_author_reconstruction", "cross_author_transfer"):
            summary = result["arm_summaries"][arm]
            bootstrap = result["book_cluster_bootstrap"][arm]
            expected = summary["expected_rows"]
            copy_failures = summary["gate_counts"]["no_copy_gate"].get("fail", 0)
            lines.append(
                f"| `{method_id}` | {ARM_LABELS[arm]} | "
                f"{fmt_number(summary['mean_target_margin'])} | "
                f"{fmt_number(summary['mean_paired_margin_lift'])} | "
                f"[{fmt_number(bootstrap['lower'])}, {fmt_number(bootstrap['upper'])}] | "
                f"{summary['positive_lift_rows']}/{expected} | "
                f"{fmt_percent(summary['target_chunk_share_scored'])} | "
                f"{summary['hard_fidelity_failure_rows']}/{expected} "
                f"({fmt_percent(summary['hard_fidelity_failure_rows'] / expected)}) | "
                f"{copy_failures}/{expected} |"
            )

    lines.extend(
        [
            "",
            "### Threshold status",
            "",
            f"The threshold artifact contains the historical value `{old_threshold:.9f}`, but its "
            f"iteration-2 status is **`{threshold['status']}`** with errors "
            f"`{', '.join(threshold['errors'])}`. Binary deterministic style success is therefore "
            "undefined in this screen. The generated `deterministic_style_success.svg` is retained "
            "for artifact completeness but must not be interpreted as a measured 0% or 80% result.",
            "",
            "The error is a binding failure, not evidence that the numeric threshold itself is "
            "scientifically wrong. Rebinding or recalibrating it now would be post-hoc because the "
            "method outputs are visible. Iteration 3 must generate a fresh threshold artifact bound "
            "to its allocation and protocol before style generation.",
            "",
            "### Promotion gate",
            "",
            "Screening required positive mean lift in both arms, no copy failure, and at most 10% "
            "hard-fidelity failures in each arm. The 24-row own-author arm permits at most two "
            "failures; the 12-row cross-author arm permits at most one.",
            "",
            "| Method | Lift positive in both arms | Own fidelity <=10% | Cross fidelity <=10% | No copy failures | Joint result |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for method_id in METHOD_ORDER[1:]:
        result = methods[method_id]
        own = result["arm_summaries"]["own_author_reconstruction"]
        cross = result["arm_summaries"]["cross_author_transfer"]
        lift_pass = (
            own["mean_paired_margin_lift"] > 0
            and cross["mean_paired_margin_lift"] > 0
        )
        own_pass = own["hard_fidelity_failure_rows"] / own["expected_rows"] <= 0.1
        cross_pass = cross["hard_fidelity_failure_rows"] / cross["expected_rows"] <= 0.1
        copy_pass = all(
            arm["gate_counts"]["no_copy_gate"].get("fail", 0) == 0
            for arm in (own, cross)
        )
        joint = lift_pass and own_pass and cross_pass and copy_pass
        lines.append(
            f"| `{method_id}` | {'pass' if lift_pass else 'fail'} | "
            f"{'pass' if own_pass else 'fail'} | {'pass' if cross_pass else 'fail'} | "
            f"{'pass' if copy_pass else 'fail'} | **{'qualifies' if joint else 'does not qualify'}** |"
        )

    lines.extend(
        [
            "",
            "### Method-level interpretation",
            "",
        ]
    )
    interpretations = {
        "aligned_pairs_only": (
            "Best evidence of stable lift: own-author +0.173 and cross-author +0.135, with both "
            "cluster bootstrap intervals above zero. It also has the strongest single cross-author "
            "gain, but 7/24 own-author rows fail the frozen fidelity screen."
        ),
        "aligned_pairs_plus_cards": (
            "Largest mean lift in both arms (+0.194 own, +0.147 cross), but cards do not solve "
            "control: 7/24 own and 4/12 cross rows fail fidelity. This arm is the strongest style "
            "mover and the weakest cross-arm fidelity result."
        ),
        "aligned_pairs_edit_plan": (
            "Lowest own-author fidelity failure rate (6/24) and an admissible 1/12 cross rate, "
            "but it still exceeds the own-arm gate. Its lift is positive and cluster-stable, yet "
            "smaller than the cards arm."
        ),
    }
    for method_id in METHOD_ORDER[1:]:
        reasons = fidelity_reasons(methods[method_id])
        reason_text = ", ".join(
            f"`{name}` ({count})" for name, count in reasons.most_common()
        ) or "none"
        lines.extend(
            [
                f"#### `{method_id}`",
                "",
                interpretations[method_id],
                "",
                f"Recorded paragraph-level surface flags: {reason_text}.",
                "",
                f"![{method_configs[method_id]['label']} paired lift]("
                f"{chart_root}/methods/{method_id}/medium/charts/paired_margin_lift.svg)",
                "",
            ]
        )

    rerank_arms = rerank["arms"] if "arms" in rerank else rerank
    lines.extend(
        [
            "## Registered Family-Rerank Diagnostic",
            "",
            "The preregistered diagnostic selects, per row, the fidelity-passing candidate with the "
            "highest target margin, then paired lift, then lexical method ID. It is not a generated "
            "method arm. Unresolved rows contribute zero lift rather than being dropped.",
            "",
            "| Arm | Rows | Selected | Unresolved | Conservative lift | Selected-only lift | Positive selected rows |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for arm in ("own_author_reconstruction", "cross_author_transfer"):
        row = rerank_arms[arm]
        lines.append(
            f"| {ARM_LABELS[arm]} | {row['rows']} | {row['selected_rows']} | "
            f"{row['unresolved_rows']} | "
            f"{fmt_number(row['conservative_mean_paired_margin_lift'])} | "
            f"{fmt_number(row['selected_only_mean_paired_margin_lift'])} | "
            f"{row['positive_lift_rows']} |"
        )
    lines.extend(
        [
            "",
            "This suggests that candidate-level selection may outperform any fixed prompt family. "
            "It does **not** establish accuracy, semantic safety, or confirmation performance. "
            "Iteration 3 must register candidate generation and reranking as the tested pipeline "
            "before output generation, with its own valid threshold and independent judgments.",
            "",
            "## Independent Evaluation",
            "",
            f"A separate `gpt-5.5` evaluator reviewed frozen aggregate files and performed "
            f"English-grounded stratified close reading. Its recorded verdict is **{audit_verdict}**. "
            f"{audit_detail}",
            "",
            "The 13 reviewed method-output cells covered every method, both arms, high- and "
            "negative-lift rows, and hard-fidelity failures. Most inspected dialogue flags were "
            "quote-attribution relocations that preserved proposition and speaker, so they look "
            "conservative for semantics. They remain valid protocol failures. The reviewer also "
            "found literal `},{` residue in some failed outputs, which is a real readability and "
            "generation-artifact defect. Inspected Latin-token mismatches generally preserved "
            "event meaning but still violated the frozen token-consistency rule.",
            "",
            "This close reading is an independent outcome audit, not the preregistered blinded "
            "semantic-judge layer. No independent judgment artifact was supplied, so semantic and "
            "readability noninferiority remain untested and cannot be claimed.",
            "",
            "The evaluator was instructed to include all three methods, both arms, high- and "
            "negative-lift cases, and deterministic fidelity failures; distinguish likely semantic "
            "risk from conservative surface false positives; and leave the preregistered gate "
            "unchanged. The full audit is at "
            "`generated/style_research/style_transfer_experiments/iterations/aligned_pairs_v1/"
            "audits/iteration2_outcome_audit.md`.",
            "",
            "## Conclusion",
            "",
            "Aligned pseudo-parallel evidence materially improves the proxy style margin compared "
            "with iteration 1's unaligned examples, and the lift is positive in both own- and "
            "cross-author arms. However, no fixed method is eligible for confirmation because its "
            "own-author fidelity failure rate exceeds the frozen 10% limit. The invalid threshold "
            "binding independently prevents any threshold-based 80% claim. **No method from "
            "iteration 2 is approved for production.**",
            "",
            "## Iteration 3 Requirements",
            "",
            "1. Freeze a correctly bound generated-domain threshold before any method output exists.",
            "2. Test the registered multi-candidate family reranker as an actual pipeline, not a post-output diagnostic.",
            "3. Add paragraph-batched long-form generation with deterministic reassembly to reduce schema retries on 46-53 paragraph chunks.",
            "4. Freeze a sanitizer that rejects literal JSON residue and a dialogue rule that preserves speech-tag placement unless explicitly allowed.",
            "5. Freeze a Latin-name/token policy and test lighter style intensity before adding more style evidence.",
            "6. Preserve the same production-compatible English-to-neutral prompt and English semantic authority.",
            "7. Run blind independent semantic/readability judgments on screened candidates before promotion.",
            "8. Open the 116-row confirmation set only if the entire frozen pipeline passes screening; require at least 80% on both registered endpoints.",
            "9. If prompting plus reranking still fails, preregister a learned adapter or preference-optimization study using the aligned pairs, then evaluate it on the same separation rules.",
            "",
            "## Reproduction",
            "",
            "Validate the frozen iteration-two package:",
            "",
            "```bash",
            "uv run python experiments/iteration2/prepare.py validate",
            "```",
            "",
            "Re-run deterministic scoring from frozen outputs:",
            "",
            "```bash",
            "uv run python experiments/iteration2/evaluate_style_transfer_methods.py evaluate \\",
            f"  --experiment-root {root.relative_to(PROJECT_ROOT)} \\",
            "  --sample-set development_proxy_v1 \\",
            "  --run-id iteration2_aligned_gpt55_v1 \\",
            f"  --analysis-lock {root.relative_to(PROJECT_ROOT)}/{active_lock} \\",
            f"  --selection-file {root.relative_to(PROJECT_ROOT)}/sample_sets/development_proxy_v1.screening_v1_ids.json \\",
            "  --scorer-mode load \\",
            "  --screening-phase initial \\",
            f"  --output-dir {root.relative_to(PROJECT_ROOT)}/evaluations/screening_v1_initial",
            "```",
            "",
            "Rebuild the family-rerank diagnostic and this report:",
            "",
            "```bash",
            "uv run python experiments/iteration2/analyze_family_rerank.py",
            "uv run python experiments/iteration2/report_style_transfer_iteration2.py",
            "```",
            "",
            "## Primary Artifacts",
            "",
            f"- Preregistration: `{root.relative_to(PROJECT_ROOT)}/protocols/iteration2_preregistration.v1.json`",
            f"- Active lock: `{root.relative_to(PROJECT_ROOT)}/{active_lock}`",
            f"- Pair-pool summary: `{root.relative_to(PROJECT_ROOT)}/sample_sets/iteration2_aligned_pair_pool_v1.summary.json`",
            f"- Proxy summary: `{root.relative_to(PROJECT_ROOT)}/sample_sets/development_proxy_v1.summary.json`",
            f"- Method registry: `{root.relative_to(PROJECT_ROOT)}/method_registry/style_methods.v1.json`",
            f"- Screening evaluation: `{root.relative_to(PROJECT_ROOT)}/evaluations/screening_v1_initial/evaluation_summary.json`",
            f"- Per-method tables and graphs: `{root.relative_to(PROJECT_ROOT)}/evaluations/screening_v1_initial/methods/`",
            f"- Family rerank: `{root.relative_to(PROJECT_ROOT)}/evaluations/screening_v1_initial/diagnostics/family_rerank/`",
            f"- Pre-generation audit: `{root.relative_to(PROJECT_ROOT)}/audits/pre_style_generation_methodology_audit.md`",
            f"- Independent outcome audit: `{root.relative_to(PROJECT_ROOT)}/audits/iteration2_outcome_audit.md`",
            "",
            "## Method Sources",
            "",
            "- Krishna, Wieting, and Iyyer (2020), STRAP: https://aclanthology.org/2020.emnlp-main.55/",
            "- Patel, Andrews, and Callison-Burch (2024 revision), Styll: https://arxiv.org/abs/2212.08986",
            "- Suzgun, Melas-Kyriazi, and Jurafsky (2022), Prompt-and-Rerank: https://aclanthology.org/2022.emnlp-main.141/",
            "- Horvitz et al. (2024), TinyStyler: https://aclanthology.org/2024.findings-emnlp.781/",
            "- Liu and May (2025), iterative preference optimization with pseudo-parallel data: https://aclanthology.org/2025.naacl-long.135/",
        ]
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = absolute(args.experiment_root)
    output = absolute(args.output)
    build_report(root, output)
    print(
        json.dumps(
            {"status": "complete", "report": str(output.relative_to(PROJECT_ROOT))},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
