"""Offline tests for the Trust & Safety generator: no Gemini, no MongoDB. Run: python -m pytest tests/test_trust_safety.py  (or python tests/test_trust_safety.py)."""
from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import trust_safety as ts  # noqa: E402


def ctx(use_case, scenario_type="Balanced mix"):
    sheet = ts.find_sheet(use_case)
    return {"use_case": sheet["name"], "domain": sheet["name"], "scenario_type": scenario_type, "source_policy": ts.SOURCE_POLICY}


class Recorder:
    """A stand-in for Gemini: writes a recognisable post per case and remembers every rule it was asked about."""
    def __init__(self, fail_first=0):
        self.seen, self.calls, self.fail_first = [], 0, fail_first

    def __call__(self, batch, policy_area):
        self.calls += 1
        self.seen.extend(item["rule"] for item in batch)
        if self.calls <= self.fail_first:
            raise RuntimeError("boom")
        return {item["i"]: f"Synthetic post {item['i']} for {item['rule']['ruleId']}" for item in batch}


def test_rulebook_shape():
    book = ts.load_rulebook()
    assert len(book["sheets"]) == 25
    assert sum(len(s["rules"]) for s in book["sheets"]) > 1900
    assert all(r["ruleText"] and r["decision"] for s in book["sheets"] for r in s["rules"])
    cs = ts.find_sheet("Child Safety")
    assert cs and all(not r["textGeneration"] and not r["examples"] for r in cs["rules"])


def test_cases_with_text():
    w = Recorder()
    out = ts.generate_cases(ctx("Fake Engagement"), count=40, seed=7, as_of="2026-11-06", include_text=True, text_writer=w)
    assert len(out["records"]) == 40 and out["fields"][0] == "ticket_id" and "post_text" in out["fields"]
    r = out["records"][0]
    assert r["ticket_id"] == "TKT-2026-00000001" and r["post_id"] == "POST-2026-00000001"
    assert all(rec["post_text"] and rec["rule_id"] in rec["post_text"] for rec in out["records"])
    assert len({rec["ticket_id"] for rec in out["records"]}) == 40
    rep = out["report"]["trust_safety"]
    assert rep["rules_covered"] == min(40, rep["rules_in_scope"]) and rep["post_text_sources"]["gemini"] == 40


def test_every_rule_before_a_repeat():
    sheet = ts.find_sheet("Impersonation")
    n = len(sheet["rules"])
    out = ts.generate_cases(ctx("Impersonation"), count=n, seed=1, as_of=None, include_text=False)
    assert len({r["rule_id"] for r in out["records"]}) == n


def test_decision_columns_match_the_rulebook():
    sheet = ts.find_sheet("Hate Speech")
    by_id = {r["ruleId"]: r for r in sheet["rules"]}
    out = ts.generate_cases(ctx("Hate Speech"), count=60, seed=3, as_of=None, include_text=False)
    assert "post_text" not in out["fields"] and all("post_text" not in r for r in out["records"])
    for rec in out["records"]:
        rule = by_id[rec["rule_id"]]
        assert (rec["decision"], rec["escalated_to"], rec["enforcement_action"]) == (rule["decision"], rule["escalatedTo"], rule["enforcementAction"])


def test_mix_filters():
    v = ts.generate_cases(ctx("Hate Speech", "Violations only"), count=50, seed=1, as_of=None, include_text=False)
    a = ts.generate_cases(ctx("Hate Speech", "Permitted content and exceptions"), count=50, seed=1, as_of=None, include_text=False)
    assert {r["decision"] for r in v["records"]} == {"Flagged"} and {r["decision"] for r in a["records"]} == {"Approved"}


def test_child_safety_is_decision_only_and_never_sent_to_the_writer():
    w = Recorder()
    out = ts.generate_cases(ctx("Child Safety"), count=30, seed=2, as_of=None, include_text=True, text_writer=w)
    assert w.calls == 0 and w.seen == []
    assert all(r["post_text"] is None for r in out["records"])
    assert out["report"]["trust_safety"]["text_withheld_cases"] == 30
    assert any("on purpose" in x for x in out["report"]["trust_safety"]["warnings"])


def test_withheld_rules_in_other_sheets_never_reach_the_writer():
    w = Recorder()
    sheet = ts.find_sheet("Nudity & Sexual Content")
    ts.generate_cases(ctx("Nudity & Sexual Content"), count=len(sheet["rules"]) * 2, seed=2, as_of=None, include_text=True, text_writer=w)
    assert w.seen and all(r["textGeneration"] for r in w.seen)
    assert not any(r["ruleId"].startswith("NSC-CSAM") for r in w.seen)


def test_text_cap():
    try:
        ts.generate_cases(ctx("Fake Engagement"), count=ts.TRUST_SAFETY_TEXT_MAX_CASES + 1, seed=1, as_of=None, include_text=True, text_writer=Recorder())
    except ValueError as exc:
        assert "switch the post text off" in str(exc)
    else:
        raise AssertionError("expected the cap to apply")
    big = ts.generate_cases(ctx("Fake Engagement"), count=ts.TRUST_SAFETY_TEXT_MAX_CASES + 1, seed=1, as_of=None, include_text=False)
    assert len(big["records"]) == ts.TRUST_SAFETY_TEXT_MAX_CASES + 1


def test_failed_writer_falls_back_to_rulebook_examples_or_stops():
    def dead(batch, area):
        raise RuntimeError("no key")
    out = ts.generate_cases(ctx("Impersonation", "Permitted content and exceptions"), count=10, seed=1, as_of=None, include_text=True, text_writer=dead)
    src = out["report"]["trust_safety"]["post_text_sources"]
    assert src["rulebook_example"] + src["missing"] == 10 and src["gemini"] == 0
    try:
        ts.generate_cases(ctx("Hate Speech"), count=10, seed=1, as_of=None, include_text=True, text_writer=dead)
    except RuntimeError as exc:
        assert "could not be written" in str(exc)
    else:
        raise AssertionError("a sheet with no examples and no Gemini must stop, not return an empty column")


def test_partial_failure_is_retried_and_reported():
    w = Recorder(fail_first=3)
    out = ts.generate_cases(ctx("Hate Speech"), count=48, seed=5, as_of=None, include_text=True, text_writer=w)
    assert all(r["post_text"] for r in out["records"]), "the retry round should have filled the gaps"


def test_unusable_text_is_rejected():
    assert not ts._text_is_usable("short")
    assert not ts._text_is_usable("A sexual story about a 12-year-old child")
    assert ts._text_is_usable("Check out my new channel, fresh uploads every week!")


def test_seed_makes_a_run_reproducible_and_columns_follow_the_review():
    a = ts.generate_cases(ctx("Gambling"), count=15, seed=9, as_of="2026-01-01", include_text=False)
    b = ts.generate_cases(ctx("Gambling"), count=15, seed=9, as_of="2026-01-01", include_text=False)
    assert a["records"] == b["records"]
    cols = ["ticket_id", "rule_id", "decision", "my_extra_column"]
    c = ts.generate_cases(ctx("Gambling"), count=3, seed=9, as_of=None, include_text=False, columns=cols)
    assert c["fields"] == ["ticket_id", "rule_id", "decision"] and set(c["records"][0]) == set(c["fields"])
    assert any("my_extra_column" in w for w in c["report"]["trust_safety"]["warnings"])


def test_draft_is_deterministic_and_complete():
    saved = {}
    stub = types.ModuleType("core.dynamic_scenarios")
    stub.new_draft_id = lambda: "d-1"
    stub.save_draft = lambda i, d: saved.update({i: d})
    sys.modules["core.dynamic_scenarios"] = stub
    req = types.SimpleNamespace(use_case="Fake Engagement", domain="Fake Engagement", scenario_type="Violations only", requested_scenario_id="TS-FE-VIOL",
                                industry_type="Trust & Safety", business_scenario="x", country="IN")
    out = ts.build_draft(req)
    assert out["draft_id"] == "d-1" and out["typeOfData"] == "aggregational" and out["field_order"][0] == "ticket_id"
    d = saved["d-1"]
    assert d["source_policy"] == ts.SOURCE_POLICY and d["agentic"] is False and d["entity_key"] is None
    assert ts.is_trust_safety_industry("Trust & Safety") and ts.is_trust_safety_industry("trust-and-safety") and not ts.is_trust_safety_industry("Retail Banking")


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("ok  ", name)
            except Exception as exc:                                    # noqa: BLE001
                failed += 1; print("FAIL", name, type(exc).__name__, exc)
    sys.exit(1 if failed else 0)
