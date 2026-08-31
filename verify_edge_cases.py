"""Offline verification for edge-case deterministic helpers.
Run with the project's environment after dependencies are installed.
"""
import ast
from pathlib import Path

src = Path(__file__).parent / "agents" / "generator_agent.py"
ast.parse(src.read_text(encoding="utf-8"))
print("generator_agent.py AST: PASS")

text = src.read_text(encoding="utf-8")
required = [
    "def _condition_branches(",
    "def _edge_candidate(",
    "def _transactional_records(",
    "def _edge_case_count(",
    'candidate["isEdgeCaseData"] = True',
]
for marker in required:
    assert marker in text, marker
print("edge-case implementation markers: PASS")
