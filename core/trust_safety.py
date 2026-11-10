"""Trust & Safety: moderation cases generated from the policy rulebook.

A Trust & Safety scenario is not described by variables with statistical generators, as telecom and banking are. Its source of
truth is a rulebook: each rule says what the content is, who decides, and what the outcome is (decision, escalation, enforcement).
One generated record is one moderation case: a rule, a synthetic post and ticket, and the outcome the rulebook prescribes for it.

  * the decision columns come straight from the rule, so they are always consistent with the rulebook;
  * the post text is optional. It is written by Gemini to fit the rule and the outcome, and is never written for the rules marked
    ``textGeneration: false`` (everything about minors in a sexual context). Those cases carry the decision columns only;
  * the rulebook is ``seed/trust_safety_rulebook.json``, built from the workbook by ``scripts/build_trust_safety_rulebook.py``.

Nothing here touches MongoDB at import time; the draft step imports the draft store only when it runs.
"""
from __future__ import annotations

import json
import logging
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable

from config.runtime import TRUST_SAFETY_TEXT_BATCH, TRUST_SAFETY_TEXT_MAX_CASES, TRUST_SAFETY_TEXT_WORKERS
from core.errors import LLMUpstreamError

logger = logging.getLogger(__name__)

SOURCE_POLICY = "trust_safety_rulebook"
VARIABLE_SOURCE = "RULEBOOK"
RULEBOOK_PATH = Path(__file__).resolve().parent.parent / "seed" / "trust_safety_rulebook.json"

# (column, dtype, description). The order is the order of the columns in the output.
CASE_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("ticket_id", "string", "Moderation ticket the case sits in (synthetic)."),
    ("post_id", "string", "Post under review (synthetic)."),
    ("policy_area", "categorical", "The policy the rule belongs to."),
    ("rule_id", "string", "Rulebook rule the case is tested against."),
    ("section_ref", "string", "Section of the policy document the rule comes from."),
    ("domain", "categorical", "Part of the rulebook: prohibited content, permitted content or exception, scoping, decision framework."),
    ("category", "categorical", "Rule category."),
    ("sub_category", "categorical", "Rule sub-category."),
    ("rule_text", "string", "The rule, as written in the rulebook."),
    ("post_text", "string", "Synthetic post written to fit the rule and its outcome. Empty where text is withheld."),
    ("decision_maker", "categorical", "Who makes the first decision (L1 reviewer)."),
    ("decision", "categorical", "The decision the rulebook gives for this case."),
    ("escalated_to", "categorical", "Who the case is escalated to, if anyone."),
    ("enforcement_action", "categorical", "The enforcement action the rulebook prescribes."),
    ("rationale", "string", "Why the rulebook gives this outcome."),
)
TEXT_COLUMN = "post_text"

MIXES = {"all": "Balanced mix of all rules", "violation": "Violations only", "allow": "Permitted content and exceptions only"}

# the same two patterns the workbook converter uses to withhold text about minors in a sexual context
_MINOR = re.compile(r"\b(minor|minors|child|children|kid|kids|underage|under-age|under 18|teen|teenager|toddler|infant|\d{1,2}[- ]year[- ]old|csam|csar|ncmec)\b", re.I)
_SEXUAL = re.compile(r"\b(sexual|sexualis|sexualiz|nude|porn|erotic|grooming|csam|csar|child abuse|abuse material|exploitation)\w*", re.I)


# ----------------------------------------------------------------------------------------------------------------------------
# rulebook
@lru_cache(maxsize=1)
def load_rulebook() -> dict[str, Any]:
    if not RULEBOOK_PATH.is_file():
        raise RuntimeError(f"The Trust & Safety rulebook is missing: {RULEBOOK_PATH.name}. Build it with scripts/build_trust_safety_rulebook.py.")
    return json.loads(RULEBOOK_PATH.read_text(encoding="utf-8"))


def is_trust_safety_industry(industry: Any) -> bool:
    text = re.sub(r"[^a-z]", "", str(industry or "").casefold())
    return text in {"trustsafety", "trustandsafety"}


def _key(text: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text or "").casefold())


def find_sheet(*names: Any) -> dict[str, Any] | None:
    """The rulebook sheet a scenario is about: matched by sheet name or id on the use case or the domain."""
    wanted = {_key(n) for n in names if n}
    for sheet in load_rulebook()["sheets"]:
        if {_key(sheet["name"]), _key(sheet["id"])} & wanted:
            return sheet
    return None


def mix_of(scenario_type: Any) -> str:
    text = str(scenario_type or "").casefold()
    if "violation" in text:
        return "violation"
    if "permitted" in text or "allow" in text:
        return "allow"
    return "all"


def rules_in_scope(sheet: dict[str, Any], mix: str) -> list[dict[str, Any]]:
    rules = sheet["rules"]
    if mix == "violation":
        return [r for r in rules if r["decisionClass"] in {"violation", "escalate"}]
    if mix == "allow":
        return [r for r in rules if r["decisionClass"] == "allow"]
    return list(rules)


def case_variables(sheet: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """The columns of a case, in the shape the review screen and /scenario/confirm expect of variables."""
    out = []
    for position, (name, dtype, description) in enumerate(CASE_COLUMNS, start=1):
        row = {"name": name, "dtype": dtype, "gen": "rulebook", "params": {}, "depends_on": [], "description": description,
               "source": VARIABLE_SOURCE, "sequence": position}
        out.append(row)
    return out


# ----------------------------------------------------------------------------------------------------------------------------
# draft (the review step)
def build_draft(req: Any) -> dict[str, Any]:
    """Create the draft for a Trust & Safety scenario, deterministically: the columns are the case columns, no model is involved.

    ``req`` is the ScenarioProposeRequest. Returns the dict the propose route answers with.
    """
    from core.dynamic_scenarios import new_draft_id, save_draft

    sheet = find_sheet(req.use_case, req.domain)
    if sheet is None:
        raise ValueError(f"There is no Trust & Safety rulebook for use case '{req.use_case}' / domain '{req.domain}'.")
    mix = mix_of(req.scenario_type)
    scope = rules_in_scope(sheet, mix)
    if not scope:
        raise ValueError(f"The '{sheet['name']}' rulebook has no rules for '{MIXES[mix]}'.")

    variables = case_variables(sheet)
    field_order = [v["name"] for v in variables]
    draft_id = new_draft_id()
    description = f"{sheet['name']} moderation cases from the Trust & Safety rulebook ({MIXES[mix].lower()}): {len(scope)} rules in scope."
    draft = {
        "label": req.requested_scenario_id, "journey": req.domain, "description": description,
        "variables": variables, "field_order": field_order, "domain": req.domain, "business_scenario": req.business_scenario,
        "business_response": None, "expected_outcome": None, "use_case": req.use_case,
        "scenario_id": req.requested_scenario_id, "requested_scenario_id": req.requested_scenario_id,
        "scenario_type": req.scenario_type, "industry_type": req.industry_type, "country": req.country,
        "type_of_data": "aggregational", "entity_key": None, "records_per_user": 10, "agentic": False,
        "source": SOURCE_POLICY, "source_policy": SOURCE_POLICY,
        "variable_sources": {v["name"]: VARIABLE_SOURCE for v in variables},
        "trust_safety": {"sheet": sheet["id"], "mix": mix, "rules_in_scope": len(scope)},
    }
    save_draft(draft_id, draft)
    return {"success": True, "draft_id": draft_id, "scenario_id": req.requested_scenario_id,
            "requested_scenario_id": req.requested_scenario_id, "journey": req.domain, "description": description,
            "variables": variables, "field_order": field_order, "typeOfData": "aggregational", "entityKey": None,
            "variableSources": {v["name"]: VARIABLE_SOURCE for v in variables}, "conceptReport": None}


# ----------------------------------------------------------------------------------------------------------------------------
# post text
TextWriter = Callable[[list[dict[str, Any]], str], dict[int, str]]

_SYSTEM = (
    "You write short synthetic example posts for a content-moderation test set. A reviewer will read each post and apply a policy rule "
    "to it, so every post must clearly fit the rule and the outcome it is given for. Rules for you: "
    "write 1 to 3 sentences in the voice of an ordinary user of a video platform (a comment, a title, a description or a short message); "
    "invent everything: no real people, brands, channels, handles or links (use example.com if a link is needed); "
    "for a post that violates the rule, show only as much as is needed for a reviewer to recognise the violation, never working instructions, "
    "quantities, methods, recipes or other operational detail; "
    "never use real slurs: write [slur] where a rule needs one; "
    "never write anything sexual involving a minor; "
    "for a post that is allowed, make it plausibly close to the rule so the reviewer has to think, but clearly acceptable under the rule. "
    "Reply with JSON only."
)


def _batch_prompt(batch: list[dict[str, Any]], policy_area: str) -> str:
    items = []
    for item in batch:
        rule = item["rule"]
        items.append({
            "i": item["i"], "category": rule.get("category"), "sub_category": rule.get("subCategory"), "rule": rule["ruleText"],
            "outcome": {"violation": "the post VIOLATES the rule", "allow": "the post is ALLOWED under the rule",
                        "escalate": "the post is borderline and must be ESCALATED"}[rule["decisionClass"]],
            "style_examples": rule.get("examples", [])[:2],
        })
    return (f"Policy: {policy_area}.\nWrite one post for each item below. Reply with a JSON array of objects "
            f'{{"i": <the item number>, "post": "<the post>"}}, one per item, nothing else.\n\n' + json.dumps(items, ensure_ascii=False))


def _text_is_usable(text: Any) -> bool:
    if not isinstance(text, str):
        return False
    text = text.strip()
    if not 8 <= len(text) <= 1500:
        return False
    return not (_MINOR.search(text) and _SEXUAL.search(text))


def gemini_text_writer(batch: list[dict[str, Any]], policy_area: str) -> dict[int, str]:
    """Ask Gemini for one post per case in ``batch``; returns the usable ones, keyed by case number."""
    reply = _client().generate_json(_SYSTEM, _batch_prompt(batch, policy_area), temperature=0.9)
    rows = reply if isinstance(reply, list) else (reply.get("posts") if isinstance(reply, dict) else None) or []
    wanted = {item["i"] for item in batch}
    out: dict[int, str] = {}
    for row in rows:
        if isinstance(row, dict) and row.get("i") in wanted and _text_is_usable(row.get("post")):
            out[int(row["i"])] = str(row["post"]).strip()
    return out


_CLIENT: dict[str, Any] = {}


def _client():
    if "c" not in _CLIENT:
        from core.llm_client import GeminiClient

        _CLIENT["c"] = GeminiClient()
    return _CLIENT["c"]


def _write_texts(cases: list[dict[str, Any]], policy_area: str, writer: TextWriter) -> tuple[dict[int, str], dict[str, int]]:
    """Post text for every case that may have it. Returns ({case number: text}, how each came about)."""
    todo = [c for c in cases if c["rule"]["textGeneration"]]
    texts: dict[int, str] = {}
    size = TRUST_SAFETY_TEXT_BATCH

    def run(batch: list[dict[str, Any]]) -> dict[int, str]:
        try:
            return writer(batch, policy_area)
        except (LLMUpstreamError, ValueError, RuntimeError, OSError) as exc:      # a provider failure, bad JSON, a missing GEMINI_API_KEY
            logger.warning("Trust & Safety text batch failed: %s", exc)
            return {}

    for _round in range(3):                                              # two retries for what came back missing or unusable
        pending = [c for c in todo if c["i"] not in texts]
        if not pending:
            break
        batches = [pending[k:k + size] for k in range(0, len(pending), size)]
        with ThreadPoolExecutor(max_workers=TRUST_SAFETY_TEXT_WORKERS) as pool:
            for result in pool.map(run, batches):
                texts.update(result)

    sources = {"gemini": len(texts), "rulebook_example": 0, "missing": 0}
    for case in todo:
        if case["i"] in texts:
            continue
        examples = [e for e in case["rule"].get("examples", []) if _text_is_usable(e)]
        if examples:                                                      # fall back to the rulebook's own example post
            texts[case["i"]] = examples[case["i"] % len(examples)]
            sources["rulebook_example"] += 1
        else:
            sources["missing"] += 1
    return texts, sources


# ----------------------------------------------------------------------------------------------------------------------------
# generation
def _pick_rules(scope: list[dict[str, Any]], count: int, rng: random.Random) -> list[dict[str, Any]]:
    """``count`` rules, every rule of the scope in turn (in a shuffled order) before any is repeated, so a small run covers many rules."""
    picked: list[dict[str, Any]] = []
    while len(picked) < count:
        order = list(scope)
        rng.shuffle(order)
        picked.extend(order[: count - len(picked)])
    return picked


def generate_cases(context: dict[str, Any], *, count: int, seed: int | None, as_of: Any, include_text: bool,
                   columns: Iterable[str] | None = None, text_writer: TextWriter | None = None) -> dict[str, Any]:
    """The cases of the scenario in ``context`` (the confirmed scenario context). Returns ``{fields, records, report}``."""
    sheet = find_sheet(context.get("use_case"), context.get("domain"))
    if sheet is None:
        raise ValueError(f"There is no Trust & Safety rulebook for use case '{context.get('use_case')}' / domain '{context.get('domain')}'.")
    mix = mix_of(context.get("scenario_type"))
    scope = rules_in_scope(sheet, mix)
    if not scope:
        raise ValueError(f"The '{sheet['name']}' rulebook has no rules for '{MIXES[mix]}'.")
    count = max(1, int(count))
    if include_text and count > TRUST_SAFETY_TEXT_MAX_CASES:
        raise ValueError(
            f"Post text is written for up to {TRUST_SAFETY_TEXT_MAX_CASES} cases per run, and {count} were requested. "
            f"Lower the count to {TRUST_SAFETY_TEXT_MAX_CASES} or fewer, or switch the post text off for this run.")

    rng = random.Random(seed)
    rules = _pick_rules(scope, count, rng)
    cases = [{"i": n, "rule": rule} for n, rule in enumerate(rules)]

    year = _year(as_of)
    warnings: list[str] = []
    texts: dict[int, str] = {}
    text_sources: dict[str, int] = {}
    if include_text:
        texts, text_sources = _write_texts(cases, sheet["name"], text_writer or gemini_text_writer)
        if text_sources.get("missing"):
            allowed = sum(1 for c in cases if c["rule"]["textGeneration"])
            if text_sources["missing"] >= allowed:
                raise RuntimeError("The post text could not be written: Gemini returned nothing usable and the rulebook has no example posts "
                                   "for these rules. Check GEMINI_API_KEY and the service logs, or switch the post text off for this run.")
            warnings.append(f"{text_sources['missing']} case(s) have no post text because Gemini returned nothing usable for them.")
    withheld = sum(1 for c in cases if not c["rule"]["textGeneration"])
    if include_text and withheld:
        warnings.append(f"{withheld} case(s) have no post text on purpose: their rules concern minors in a sexual context, so only the decision columns are produced.")

    records = []
    for case in cases:
        rule, n = case["rule"], case["i"] + 1
        records.append({
            "ticket_id": f"TKT-{year}-{n:08d}", "post_id": f"POST-{year}-{n:08d}", "policy_area": sheet["name"], "rule_id": rule["ruleId"],
            "section_ref": rule.get("sectionRef"), "domain": rule.get("domain"), "category": rule.get("category"),
            "sub_category": rule.get("subCategory"), "rule_text": rule["ruleText"], TEXT_COLUMN: texts.get(case["i"]) if include_text else None,
            "decision_maker": rule.get("decisionMaker"), "decision": rule.get("decision"), "escalated_to": rule.get("escalatedTo"),
            "enforcement_action": rule.get("enforcementAction"), "rationale": rule.get("rationale"),
        })

    known = [name for name, _t, _d in CASE_COLUMNS]
    fields = [name for name in (columns if columns is not None else known) if name in known]
    unknown = [name for name in (columns or []) if name not in known]
    if unknown:
        warnings.append("Columns that are not part of the rulebook were left out: " + ", ".join(unknown) + ".")
    if not include_text:
        fields = [f for f in fields if f != TEXT_COLUMN]
    records = [{f: row.get(f) for f in fields} for row in records]

    report = {
        "generation_engine": SOURCE_POLICY, "quality_target_met": True, "valid_record_rate": 100.0,
        "trust_safety": {
            "policy_area": sheet["name"], "mix": MIXES[mix], "cases": count, "rules_in_scope": len(scope),
            "rules_covered": len({r["ruleId"] for r in rules}), "post_text": bool(include_text),
            "post_text_sources": text_sources, "text_withheld_cases": withheld if include_text else None,
            "decisions": dict(Counter(r["decisionClass"] for r in rules)), "warnings": warnings,
        },
    }
    return {"fields": fields, "records": records, "report": report}


def _year(as_of: Any) -> int:
    try:
        return datetime.fromisoformat(str(as_of).replace("Z", "+00:00")).year if as_of else datetime.now(timezone.utc).year
    except ValueError:
        return datetime.now(timezone.utc).year
