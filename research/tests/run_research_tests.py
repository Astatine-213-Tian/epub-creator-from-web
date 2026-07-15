from __future__ import annotations

"""Run the portable research regression suite from the research root."""

import sys
import unittest
from collections.abc import Iterator
from pathlib import Path


RESEARCH_ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_TESTS = {
    (
        "tests.experiments.iteration5.test_cr_fysm_v4_construct_v2."
        "ConstructV2ProtocolTests."
        "test_preregistered_and_executed_command_templates_match"
    ): (
        "requires the exact native Codex installation captured by the frozen "
        "construct-v2 preregistration"
    ),
}


def iter_cases(suite: unittest.TestSuite) -> Iterator[unittest.TestCase]:
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            yield from iter_cases(test)
        else:
            yield test


def main() -> int:
    discovered = unittest.defaultTestLoader.discover(
        str(RESEARCH_ROOT / "tests"),
        pattern="test_*.py",
        top_level_dir=str(RESEARCH_ROOT),
    )
    portable = unittest.TestSuite()
    excluded: list[tuple[str, str]] = []
    for test in iter_cases(discovered):
        test_id = test.id()
        reason = EXCLUDED_TESTS.get(test_id)
        if reason is None:
            portable.addTest(test)
        else:
            excluded.append((test_id, reason))

    for test_id, reason in excluded:
        print(f"[environment-bound] {test_id}: {reason}")

    result = unittest.TextTestRunner(verbosity=1).run(portable)
    if set(test_id for test_id, _reason in excluded) != set(EXCLUDED_TESTS):
        print("configured environment-bound test was not discovered", file=sys.stderr)
        return 2
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
