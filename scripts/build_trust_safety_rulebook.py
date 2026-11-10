"""Convert the Trust & Safety rulebook workbook into seed/trust_safety_rulebook.json.

Developer tool, not used at run time (needs `pip install openpyxl`). It also adds (or refreshes) the Trust & Safety industry in
seed/scenario_catalog.json, which the service inserts into MongoDB at start-up if the entries are missing.

    python scripts/build_trust_safety_rulebook.py <rulebook.xlsx> [seed/trust_safety_rulebook.json]

What it handles, because the workbook is not uniform:
  * the header text differs between sheets ("Post ID (synthetic)", "Post ID\\n(to be generated synthetically)", "TicketID", ...);
  * one sheet (Impersonation) holds a second block with the columns in a different order, under its own header row;
  * Excel cuts sheet names at 31 characters, so full names come from SHEET_INFO;
  * a Rule_ID repeats once per example post, so rows are grouped into rules that each carry a list of example posts;
  * wording differs between sheets for the same thing ("Strike - 1st" vs "Strike — 1st", "None" vs blank, "Violates Policy" vs "Flagged").
Rules about minors in a sexual context, and the whole Child Safety sheet, are marked textGeneration=false and carry no example posts,
so the generator can only produce decision columns for them.
"""
from __future__ import annotations

import json
import re
import sys
from collections import OrderedDict
from pathlib import Path

import openpyxl

# sheet title in the workbook -> (full name, policy family)
SHEET_INFO = OrderedDict([
    ("Fake Engagement", ("Fake Engagement", "Spam & Deceptive Practices")),
    ("Impersonation", ("Impersonation", "Spam & Deceptive Practices")),
    ("Misleading Metadata", ("Misleading Metadata", "Spam & Deceptive Practices")),
    ("External Links Policy", ("External Links Policy", "Spam & Deceptive Practices")),
    ("Scams & Financially Deceptive C", ("Scams & Financially Deceptive Content", "Spam & Deceptive Practices")),
    ("Previously Removed Content", ("Previously Removed Content", "Spam & Deceptive Practices")),
    ("Child Safety", ("Child Safety", "Child Safety")),
    ("Nudity & Sexual Content", ("Nudity & Sexual Content", "Sensitive Content")),
    ("Self-Harm & Suicide", ("Self-Harm & Suicide", "Sensitive Content")),
    ("Eating Disorders", ("Eating Disorders", "Sensitive Content")),
    ("Violent Extremism & Terrorism", ("Violent Extremism & Terrorism", "Violent & Hateful Content")),
    ("Hate Speech", ("Hate Speech", "Violent & Hateful Content")),
    ("Harassment & Cyberbullying", ("Harassment & Cyberbullying", "Violent & Hateful Content")),
    ("Harmful or Dangerous Content", ("Harmful or Dangerous Content", "Violent & Hateful Content")),
    ("Violent Criminal Organizations", ("Violent Criminal Organizations", "Violent & Hateful Content")),
    ("Medical & Health Misinformation", ("Medical & Health Misinformation", "Misinformation")),
    ("Electoral & Civic Misinformatio", ("Electoral & Civic Misinformation", "Misinformation")),
    ("Conspiracy Theories", ("Conspiracy Theories", "Misinformation")),
    ("Firearms & Weapons", ("Firearms & Weapons", "Regulated Goods & Services")),
    ("Drugs & Controlled Substances", ("Drugs & Controlled Substances", "Regulated Goods & Services")),
    ("Alcohol & Tobacco", ("Alcohol & Tobacco", "Regulated Goods & Services")),
    ("Gambling", ("Gambling", "Regulated Goods & Services")),
    ("Counterfeit & Fraudulent Goods", ("Counterfeit & Fraudulent Goods", "Regulated Goods & Services")),
    ("Doxxing", ("Doxxing", "Privacy & Intellectual Property")),
    ("Copyright & Intellectual Proper", ("Copyright & Intellectual Property", "Privacy & Intellectual Property")),
])

# the whole sheet is decision-only (no synthetic post text)
TEXT_DISABLED_SHEETS = {"Child Safety"}

FIELDS = {
    "ruleid": "rule_id", "sectionref": "section_ref", "postid": "post_id", "ticketid": "ticket_id",
    "preset": "preset", "domain": "domain", "category": "category", "subcategory": "sub_category",
    "coredefinitionorrule": "rule_text", "freetext": "example_text",
    "primarydecisionmaker": "decision_maker", "decision": "decision",
    "escalatedto": "escalated_to", "escalationto": "escalated_to",
    "enforcementaction": "enforcement_action", "rationale": "rationale",
}

MINOR_RE = re.compile(r"\b(minor|minors|child|children|kid|kids|underage|under-age|under 18|teen|teenager|toddler|infant|\d{1,2}[- ]year[- ]old|csam|csar|ncmec)\b", re.I)
SEXUAL_RE = re.compile(r"\b(sexual|sexualis|sexualiz|nude|porn|erotic|grooming|csam|csar|child abuse|abuse material|exploitation)\w*", re.I)


def norm_header(value) -> str:
    # "Free Text\n(to be generated synthetically)" -> "freetext"; "Post ID (synthetic)" -> "postid"
    text = re.sub(r"\(.*?\)", "", str(value or ""), flags=re.S)
    return re.sub(r"[^a-z]", "", text.lower())


def clean(value):
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value).replace("—", "-").replace("–", "-")).strip()
    return None if text == "" or text.lower() in {"none", "n/a", "na", "nan"} else text


def decision_class(decision: str | None, action: str | None) -> str:
    d = (decision or "").lower()
    if d.startswith("escalat") or d in {"borderline"}:
        return "escalate"
    if d in {"approved", "compliant"} or d.startswith("compliant"):
        return "allow"
    return "violation"


def iter_blocks(ws):
    header = None
    for row in ws.iter_rows(values_only=True):
        if not any(c not in (None, "") for c in row):
            continue
        if str(row[0] or "").strip().lower() == "rule_id":
            header = {i: FIELDS[norm_header(c)] for i, c in enumerate(row) if c and norm_header(c) in FIELDS}
            continue
        if header is None:
            continue
        yield {name: clean(row[i]) for i, name in header.items() if i < len(row)}


def build(xlsx: Path) -> dict:
    wb = openpyxl.load_workbook(xlsx, data_only=True)
    sheets, problems = [], []
    for title in wb.sheetnames:
        if title not in SHEET_INFO:
            problems.append(f"sheet '{title}' is not in SHEET_INFO and was skipped")
            continue
        name, family = SHEET_INFO[title]
        text_off_sheet = title in TEXT_DISABLED_SHEETS
        rules: "OrderedDict[str, dict]" = OrderedDict()
        for r in iter_blocks(wb[title]):
            rid = r.get("rule_id")
            if not rid or not r.get("rule_text"):
                continue
            blob = " ".join(filter(None, [r.get("rule_text"), r.get("category"), r.get("sub_category"), r.get("escalated_to")]))
            minors_sexual = bool(MINOR_RE.search(blob) and SEXUAL_RE.search(blob))
            text_off = text_off_sheet or minors_sexual
            rule = rules.get(rid)
            if rule is None:
                rule = rules[rid] = {
                    "ruleId": rid, "sectionRef": r.get("section_ref"), "domain": r.get("domain"),
                    "category": r.get("category"), "subCategory": r.get("sub_category"), "ruleText": r.get("rule_text"),
                    "decisionMaker": r.get("decision_maker") or "L1", "decision": r.get("decision"),
                    "decisionClass": decision_class(r.get("decision"), r.get("enforcement_action")),
                    "escalatedTo": r.get("escalated_to"), "enforcementAction": r.get("enforcement_action"),
                    "rationale": r.get("rationale"), "textGeneration": not text_off, "examples": [],
                }
            else:
                same = (rule["decision"], rule["escalatedTo"], rule["enforcementAction"]) == (
                    r.get("decision"), r.get("escalated_to"), r.get("enforcement_action"))
                if not same:
                    problems.append(f"{title}: rule {rid} repeats with a different outcome; the first row was kept")
                rule["textGeneration"] = rule["textGeneration"] and not text_off
            if r.get("example_text") and not text_off and r["example_text"] not in rule["examples"]:
                rule["examples"].append(r["example_text"])
        for rule in rules.values():
            if not rule["textGeneration"]:
                rule["examples"] = []
        sheets.append({
            "id": re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-"),
            "name": name, "family": family, "sourceSheet": title,
            "textGeneration": not text_off_sheet, "rules": list(rules.values()),
        })
    return {"version": 1, "source": xlsx.name, "problems": problems, "sheets": sheets}


# ---- catalog ---------------------------------------------------------------------------------------------------------------
INDUSTRY_ID = "trust-safety"
INDUSTRY_NAME = "Trust & Safety"
SHEET_CODES = {
    "fake-engagement": "FE", "impersonation": "IMP", "misleading-metadata": "MM", "external-links-policy": "EL",
    "scams-financially-deceptive-content": "SCAM", "previously-removed-content": "PRC", "child-safety": "CS",
    "nudity-sexual-content": "NSC", "self-harm-suicide": "SHS", "eating-disorders": "ED", "violent-extremism-terrorism": "VET",
    "hate-speech": "HS", "harassment-cyberbullying": "HR", "harmful-or-dangerous-content": "HDC", "violent-criminal-organizations": "VCO",
    "medical-health-misinformation": "MED", "electoral-civic-misinformation": "ELEC", "conspiracy-theories": "CON",
    "firearms-weapons": "FW", "drugs-controlled-substances": "DRUG", "alcohol-tobacco": "AT", "gambling": "GAM",
    "counterfeit-fraudulent-goods": "CFG", "doxxing": "DOX", "copyright-intellectual-property": "IP",
}
FAMILY_BLURB = {
    "Spam & Deceptive Practices": "Fake engagement, impersonation, misleading metadata, external links, scams and re-uploads of removed content",
    "Child Safety": "Absolute prohibitions, age-restricted and educational context, and family content involving minors",
    "Sensitive Content": "Nudity and sexual content, self-harm and suicide, and eating disorders",
    "Violent & Hateful Content": "Violent extremism, hate speech, harassment, dangerous content and violent criminal organisations",
    "Misinformation": "Medical and health, electoral and civic, and conspiracy-theory misinformation",
    "Regulated Goods & Services": "Firearms, drugs, alcohol and tobacco, gambling and counterfeit goods",
    "Privacy & Intellectual Property": "Doxxing and copyright",
}
MIX_TYPES = (("MIX", "Balanced mix", "every rule of the policy, so violations, permitted content and exceptions all appear"),
             ("VIOL", "Violations only", "only the rules whose outcome is a violation or an escalation"),
             ("ALLOW", "Permitted content and exceptions", "only the rules whose outcome is that the content is allowed"))


def _in_scope(sheet: dict, key: str) -> int:
    rules = sheet["rules"]
    if key == "VIOL":
        return sum(1 for r in rules if r["decisionClass"] in {"violation", "escalate"})
    if key == "ALLOW":
        return sum(1 for r in rules if r["decisionClass"] == "allow")
    return len(rules)


def build_catalog(rulebook: dict) -> dict:
    """The Trust & Safety industry as a catalog import body: policy family -> policy (use case) -> one journey -> three scenarios."""
    families: "OrderedDict[str, list]" = OrderedDict()
    for sheet in rulebook["sheets"]:
        families.setdefault(sheet["family"], []).append(sheet)
    subs = []
    for family, sheets in families.items():
        cases = []
        for sheet in sheets:
            code = SHEET_CODES[sheet["id"]]
            scenarios = []
            for key, label, what in MIX_TYPES:
                n = _in_scope(sheet, key)
                if n == 0:
                    continue
                scenarios.append({
                    "scenarioId": f"TS-{code}-{key}", "scenarioType": label,
                    "businessScenario": (f"Generate moderation cases for the {sheet['name']} policy, drawn from {what} ({n} rules). "
                                         "Each case pairs a rule with a synthetic post and ticket and the outcome the rulebook gives it: "
                                         "decision, escalation and enforcement action.\n\n"
                                         "Business response: Give reviewers, QA and classifier teams a realistic, rulebook-consistent set of cases to train on, "
                                         "calibrate against and test.\n\n"
                                         "Expected outcome: A labelled set of moderation cases whose decisions can be traced to a rule and section of the policy."),
                    "description": f"{label}: {n} {sheet['name']} rules.",
                })
            cases.append({
                "name": sheet["name"], "proposeValue": sheet["name"],
                "description": f"Moderation cases for the {sheet['name']} policy, generated from the rulebook.",
                "journeys": [{"name": "Moderation cases", "dataTypes": ["aggregational"], "industryType": INDUSTRY_NAME,
                              "domain": sheet["name"], "scenarios": scenarios}],
            })
        subs.append({"name": family, "description": FAMILY_BLURB.get(family, family), "useCases": cases})
    return {"id": INDUSTRY_ID, "name": INDUSTRY_NAME,
            "description": "Content-moderation policy rulebooks: rules, decisions, escalations and enforcement", "subDomains": subs}


def merge_catalog(rulebook: dict, catalog_path: Path) -> None:
    data = json.loads(catalog_path.read_text(encoding="utf-8"))
    industries = [i for i in data["industries"] if i.get("id") != INDUSTRY_ID]
    industries.append(build_catalog(rulebook))
    data["industries"] = industries
    catalog_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent.parent / "seed" / "trust_safety_rulebook.json"
    data = build(src)
    dst.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    merge_catalog(data, dst.parent / "scenario_catalog.json")
    for s in data["sheets"]:
        n_ex = sum(len(r["examples"]) for r in s["rules"])
        off = sum(1 for r in s["rules"] if not r["textGeneration"])
        print(f"{s['name']:42} {s['family']:32} rules={len(s['rules']):4} examples={n_ex:4} decision-only={off}")
    for p in data["problems"]:
        print("NOTE:", p)
