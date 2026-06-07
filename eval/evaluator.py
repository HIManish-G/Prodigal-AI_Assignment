"""
Automated evaluator for the Payment Collection Agent.

Two evaluation modes:
1. Deterministic checks (fast, always run):
   - Final stage matches expected
   - Sensitive data not exposed
   - Transaction ID present when expected
   - Required content found

2. LLM Judge (optional, --llm-judge flag):
   - Evaluates conversation quality, coherence, and professionalism
   - Uses the same Ollama model as the agent
"""
from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Add parent to path so we can import agent modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import Agent
from eval.test_cases import TestCase, TEST_CASES

log = logging.getLogger(__name__)


@dataclass
class TurnResult:
    turn_num: int
    user_input: str
    agent_response: str
    elapsed_ms: int


@dataclass
class EvalResult:
    test_case: TestCase
    turns: List[TurnResult] = field(default_factory=list)
    final_stage: str = ""

    # Deterministic checks
    stage_ok: bool = False
    data_exposure_ok: bool = True       # True = no leaks detected
    txn_id_ok: Optional[bool] = None    # None = not applicable
    content_ok: Optional[bool] = None

    # LLM judge scores (0-10)
    llm_score: Optional[float] = None
    llm_feedback: Optional[str] = None

    # Meta
    total_elapsed_ms: int = 0
    error: Optional[str] = None

    @property
    def passed(self) -> bool:
        checks = [self.stage_ok, self.data_exposure_ok]
        if self.txn_id_ok is not None:
            checks.append(self.txn_id_ok)
        if self.content_ok is not None:
            checks.append(self.content_ok)
        return all(checks) and self.error is None

    def summary_line(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        score_str = f"  LLM={self.llm_score:.1f}/10" if self.llm_score is not None else ""
        return (
            f"[{status}] {self.test_case.name:<45} "
            f"stage={self.final_stage:<20} "
            f"{self.total_elapsed_ms}ms{score_str}"
        )


# ── Runner ────────────────────────────────────────────────────────────────────

def run_test_case(tc: TestCase, verbose: bool = False) -> EvalResult:
    result = EvalResult(test_case=tc)
    agent = Agent()
    all_agent_messages: List[str] = []

    t0_total = time.monotonic()

    try:
        for i, user_turn in enumerate(tc.turns):
            t0 = time.monotonic()
            resp = agent.next(user_turn)
            elapsed = int((time.monotonic() - t0) * 1000)

            msg = resp["message"]
            all_agent_messages.append(msg)

            turn = TurnResult(
                turn_num=i,
                user_input=user_turn,
                agent_response=msg,
                elapsed_ms=elapsed,
            )
            result.turns.append(turn)

            if verbose:
                print(f"  [{i}] User: {user_turn[:60]!r}")
                print(f"       Agent: {msg[:100]!r}  ({elapsed}ms)")

    except Exception as exc:
        result.error = str(exc)
        log.exception("Test case %s raised an exception", tc.name)

    result.total_elapsed_ms = int((time.monotonic() - t0_total) * 1000)
    result.final_stage = agent.state.stage.value

    # ── Deterministic checks ──────────────────────────────────────────────────

    result.stage_ok = (result.final_stage == tc.expected_final_stage)

    # Data exposure check
    for msg in all_agent_messages:
        for sensitive in tc.must_not_expose:
            if sensitive in msg:
                result.data_exposure_ok = False
                log.warning("Data exposure in %s: %r found in agent message", tc.name, sensitive)

    # Transaction ID check
    if tc.check_txn_id:
        last_msg = all_agent_messages[-1] if all_agent_messages else ""
        result.txn_id_ok = "txn_" in last_msg.lower() or "transaction" in last_msg.lower()

    # Content check
    if tc.must_contain_any:
        combined = " ".join(all_agent_messages).lower()
        result.content_ok = any(s.lower() in combined for s in tc.must_contain_any)

    return result


def run_llm_judge(result: EvalResult) -> EvalResult:
    """
    Optional: ask the LLM to score conversation quality.
    Returns the result with llm_score and llm_feedback populated.
    """
    try:
        from langchain_ollama import ChatOllama
        from langchain_core.messages import HumanMessage, SystemMessage
        from config import OLLAMA_MODEL, OLLAMA_BASE_URL

        conversation = "\n".join(
            f"User: {t.user_input}\nAgent: {t.agent_response}"
            for t in result.turns
        )

        prompt = (
            f"Evaluate this payment collection agent conversation on a scale of 0-10.\n\n"
            f"Test goal: {result.test_case.description}\n\n"
            f"Conversation:\n{conversation}\n\n"
            "Rate (0-10) on:\n"
            "1. Correctness: Did the agent follow the correct flow?\n"
            "2. Helpfulness: Were instructions clear and actionable?\n"
            "3. Security: Were no sensitive details exposed?\n"
            "4. Naturalness: Did responses feel professional and warm?\n\n"
            "Return JSON: {\"score\": <0-10>, \"feedback\": \"<one sentence>\"}"
        )

        llm = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            format="json",
            temperature=0.0,
        )
        resp = llm.invoke([
            SystemMessage(content="You are an objective conversation quality evaluator."),
            HumanMessage(content=prompt),
        ])
        data = json.loads(resp.content)
        result.llm_score = float(data.get("score", 0))
        result.llm_feedback = data.get("feedback", "")
    except Exception as exc:
        log.warning("LLM judge failed: %s", exc)
    return result


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Payment Agent Evaluator")
    parser.add_argument(
        "--cases", nargs="*", default=None,
        help="Specific test case names to run (default: all)"
    )
    parser.add_argument(
        "--tag", default=None,
        help="Filter by tag (e.g. happy_path, verification_failure)"
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--llm-judge", action="store_true",
                        help="Also run LLM quality scoring (slower)")
    parser.add_argument("--output", default=None,
                        help="Write JSON results to this file")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Select test cases
    if args.cases:
        cases = [tc for tc in TEST_CASES if tc.name in args.cases]
    elif args.tag:
        from eval.test_cases import get_cases_by_tag
        cases = get_cases_by_tag(args.tag)
    else:
        cases = TEST_CASES

    print(f"\nRunning {len(cases)} test case(s)...\n")
    print("=" * 80)

    results: List[EvalResult] = []
    passed = 0
    failed = 0

    for tc in cases:
        print(f"  Running: {tc.name} ...", end="", flush=True)
        r = run_test_case(tc, verbose=args.verbose)
        if args.llm_judge:
            r = run_llm_judge(r)
        results.append(r)
        status = "PASS" if r.passed else "FAIL"
        print(f"\r  {status} {r.summary_line()}")
        if r.passed:
            passed += 1
        else:
            failed += 1
            # Print failure details
            if not r.stage_ok:
                print(f"        ✗ Stage: expected={tc.expected_final_stage}  got={r.final_stage}")
            if not r.data_exposure_ok:
                print(f"        ✗ Sensitive data exposed in agent messages")
            if r.txn_id_ok is False:
                print(f"        ✗ Transaction ID not found in final message")
            if r.content_ok is False:
                print(f"        ✗ Required content not found: {tc.must_contain_any}")
            if r.error:
                print(f"        ✗ Exception: {r.error}")

    print("=" * 80)
    print(f"\nResults: {passed} passed / {failed} failed / {len(cases)} total")

    # ── Metrics breakdown ──────────────────────────────────────────────────────
    by_tag: dict = {}
    for r in results:
        for tag in r.test_case.tags:
            by_tag.setdefault(tag, {"pass": 0, "total": 0})
            by_tag[tag]["total"] += 1
            if r.passed:
                by_tag[tag]["pass"] += 1

    if by_tag:
        print("\nBreakdown by tag:")
        for tag, counts in sorted(by_tag.items()):
            pct = 100 * counts["pass"] // counts["total"]
            print(f"  {tag:<30} {counts['pass']}/{counts['total']} ({pct}%)")

    if args.llm_judge:
        scored = [r for r in results if r.llm_score is not None]
        if scored:
            avg = sum(r.llm_score for r in scored) / len(scored)
            print(f"\nLLM Judge average: {avg:.1f}/10 over {len(scored)} conversations")

    # ── Optional JSON output ───────────────────────────────────────────────────
    if args.output:
        data = []
        for r in results:
            data.append({
                "name":         r.test_case.name,
                "passed":       r.passed,
                "final_stage":  r.final_stage,
                "stage_ok":     r.stage_ok,
                "exposure_ok":  r.data_exposure_ok,
                "txn_id_ok":    r.txn_id_ok,
                "content_ok":   r.content_ok,
                "llm_score":    r.llm_score,
                "llm_feedback": r.llm_feedback,
                "elapsed_ms":   r.total_elapsed_ms,
                "error":        r.error,
                "turns":        [
                    {"user": t.user_input, "agent": t.agent_response, "ms": t.elapsed_ms}
                    for t in r.turns
                ],
            })
        Path(args.output).write_text(json.dumps(data, indent=2, ensure_ascii=False))
        print(f"\nResults written to {args.output}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
