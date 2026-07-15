#!/usr/bin/env python3
from __future__ import annotations

import re
import unittest

from experiments.validation.construct import prepare_cr_fysm_v4_construct_v2 as prepare
from experiments.validation.construct import run_cr_fysm_v4_construct_generation as runner
from experiments.validation.construct.construct_v2_protocol import (
    deterministic_blocks,
    han_ngram_overlap,
    outcome_protocol,
    stable_side_order,
)


class ConstructV2ProtocolTests(unittest.TestCase):
    def source(self, text: str) -> dict[str, object]:
        return {"text": text, "chunk_clean_cjk_count": 1200}

    def test_sample_id_matches_reused_schema_contract(self) -> None:
        sample_id = "s_" + prepare.stable_hash("construct-v2:test")[:24]
        self.assertRegex(sample_id, re.compile(r"^s_[0-9a-f]{24}$"))

    def test_source_cleanliness_rejects_known_residue(self) -> None:
        clean_lines = [f"这是用于协议测试的正常中文段落内容第{i}段。" for i in range(10)]
        clean = "\n".join(clean_lines)
        self.assertTrue(prepare.safe_source(self.source(clean)))
        for residue in (
            "正文完",
            "全文完",
            "本章完",
            "全书完",
            "本文到此彻底完结",
            "最后的番外",
            "《测试书名》",
            "作者：测试作者",
            "作者有话说：感谢大家",
            "作者有话要说：感谢大家",
            "作者的话：感谢大家",
            "作话：感谢大家",
            "***契合正文的番外已补，在作者有话说。",
            "这是我写过的幻想线，直接拉末尾看番外。",
            "身为作者的我想说几句",
            "——中秋番外：测试 · 完——",
            "——番外：测试：终——",
            "——测试书名·End——",
            "写序非我所长",
            "关于内容，我不愿剧透太多",
            "内容标签：强强 穿越时空",
            "搜索关键字：主角：甲",
            "一句话简介：测试",
            "立意：测试",
            "文案已于2022.9.27截图",
            "?",
            "□",
            "�",
            "€€€€",
            "正文。(ｍｉｄｕｘｓ)(ｃｏｍ)",
            "正文。ℂ()ℂ✓来ℂ笔?╬ℂ()•(ｃｏｍ)",
            "第五章 测试",
            "番外1 画室",
            "请不要把我的文和其他作者的文进行比较，双方读者都会尴尬。",
            "请不要安利我的文，公告和文案里已经说过了。",
        ):
            with self.subTest(residue=residue):
                self.assertFalse(prepare.safe_source(self.source(clean + "\n" + residue)))
        duplicated = clean + "\n重复的一行。\n重复的一行。"
        self.assertFalse(prepare.safe_source(self.source(duplicated)))
        explicit = clean + "\n他触碰对方的穴口并继续套弄。"
        self.assertFalse(prepare.safe_source(self.source(explicit)))
        for explicit_context in (
            "他进入，强袭，用胸膛压着对方，仿佛在侵略自己。",
            "床板吱呀吱呀响个不停，直至那男人办完事。",
            "他推荐了淫唐传，还问有没有高H版本。",
            "他浑身潮红，在喘息时腰也被掐红了。",
        ):
            with self.subTest(explicit_context=explicit_context):
                self.assertFalse(
                    prepare.safe_source(self.source(clean + "\n" + explicit_context))
                )
        for chunk_id in prepare.KNOWN_UNSAFE_SOURCE_CHUNK_IDS:
            with self.subTest(chunk_id=chunk_id):
                source = self.source(clean)
                source["chunk_id"] = chunk_id
                self.assertFalse(prepare.safe_source(source))

    def test_reference_copy_gate_covers_both_example_sides(self) -> None:
        examples = [
            {
                "pair_id": "pair-1",
                "neutral_zh": "甲乙丙丁戊己庚辛壬癸",
                "target_style_zh": "子丑寅卯辰巳午未申酉",
            }
        ]
        self.assertEqual(
            runner.copied_reference_ids("前文甲乙丙丁戊己庚辛后文", examples),
            ["pair-1:neutral_zh"],
        )
        self.assertEqual(
            runner.copied_reference_ids("前文子丑寅卯辰巳午未后文", examples),
            ["pair-1:target_style_zh"],
        )
        self.assertEqual(runner.copied_reference_ids("完全不同的候选文本", examples), [])

    def test_aggregate_reference_overlap_is_measured(self) -> None:
        references = ["甲乙丙丁戊己庚辛", "天地玄黄宇宙洪荒"]
        self.assertGreater(han_ngram_overlap("前文甲乙丙丁后文", references, n=4), 0.0)
        self.assertEqual(han_ngram_overlap("完全不同的候选文本", references, n=4), 0.0)

    def test_deterministic_blocks_are_contiguous_complete_and_stable(self) -> None:
        paragraphs = [
            {"id": f"p{index:04d}", "zh": "正文" * (index + 2)}
            for index in range(1, 11)
        ]
        first = deterministic_blocks(paragraphs)
        second = deterministic_blocks(paragraphs)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)
        self.assertEqual([value for block in first for value in block], [row["id"] for row in paragraphs])

    def test_rater_side_order_is_stable_and_rater_specific(self) -> None:
        value = stable_side_order(pair_id="pair-1", rater_id="rater-a", seed=20260717)
        self.assertEqual(
            value,
            stable_side_order(pair_id="pair-1", rater_id="rater-a", seed=20260717),
        )
        self.assertIn(value, {"style_first", "control_first"})

    def test_outcome_protocol_is_fixed_before_generation(self) -> None:
        protocol = outcome_protocol()
        self.assertEqual(protocol["status"], "fixed_before_any_construct_generation")
        self.assertEqual(protocol["units"]["matched_blocks"], 250)
        self.assertEqual(protocol["generation_closure"]["required_stage_artifacts"], 250)
        self.assertTrue(protocol["semantic_validation"]["all_50_sources_must_meet_minimum"])

    def test_schema_validation_rejects_missing_and_extra_fields(self) -> None:
        schema = prepare.read_json(prepare.STYLE_SCHEMA)
        valid = {
            "sample_id": "s_" + "a" * 24,
            "paragraphs": [{"id": "p0001", "zh": "有效文本"}],
            "style_cues_applied": [],
            "uncertainties": [],
        }
        self.assertEqual(runner.validate_schema_instance(valid, schema), [])
        missing = dict(valid)
        missing.pop("style_cues_applied")
        self.assertTrue(
            any("missing required property style_cues_applied" in error for error in runner.validate_schema_instance(missing, schema))
        )
        extra = {**valid, "unregistered": True}
        self.assertTrue(
            any("unexpected property unregistered" in error for error in runner.validate_schema_instance(extra, schema))
        )

    def test_english_validation_rejects_cjk_leakage(self) -> None:
        sample_id = "s_" + "b" * 24
        request = {
            "sample_id": sample_id,
            "paragraphs": [{"id": "p0001", "zh": "原文", "draft_en": "Draft."}],
        }
        preregistration = {
            "generation_stages": {
                "english_adjudication": {"schema_path": str(prepare.ADJUDICATION_SCHEMA)}
            }
        }
        for leaked_text in (
            "原文",
            "compatibility 神",
            "supplementary 𠀀",
            "radical ⼀",
            "iteration 々",
            "zero 〇",
            "Hangzhou numeral 〡",
            "vertical iteration 〻",
            "old Chinese mark 𖿢",
        ):
            with self.subTest(leaked_text=leaked_text):
                result = {
                    "sample_id": sample_id,
                    "approved": True,
                    "paragraphs": [{"id": "p0001", "en": leaked_text}],
                    "issues": [],
                }
                errors = runner.validate_result(
                    "english_adjudication",
                    request,
                    result,
                    preregistration,
                )
                self.assertIn("English output contains CJK characters", errors)

    def test_han_script_table_is_sorted_nonoverlapping_and_complete(self) -> None:
        previous_upper = -1
        codepoint_count = 0
        for lower, upper in runner.HAN_SCRIPT_RANGES:
            self.assertGreater(lower, previous_upper)
            self.assertGreaterEqual(upper, lower)
            previous_upper = upper
            codepoint_count += upper - lower + 1
        self.assertEqual(codepoint_count, 103_351)

    def test_preregistered_and_executed_command_templates_match(self) -> None:
        prepared = prepare.codex_runtime_provenance()
        current = runner.current_codex_runtime()
        self.assertEqual(prepared["codex_wrapper_resolved"], current["codex_wrapper_resolved"])
        self.assertEqual(prepared["codex_wrapper_sha256"], current["codex_wrapper_sha256"])
        self.assertEqual(prepared["codex_executable_resolved"], current["codex_executable_resolved"])
        self.assertEqual(prepared["codex_executable_sha256"], current["codex_executable_sha256"])
        self.assertEqual(prepared["codex_version"], current["codex_version"])
        self.assertEqual(prepared["command_template"], current["command_template"])
        self.assertEqual(
            prepared["command_template_sha256"],
            current["command_template_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
