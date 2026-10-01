"""Application service for scenario variable persistence.

MongoDB collection access lives in the models package; this module keeps the
business-facing service API used by the workflow and HTTP layer.
"""
from __future__ import annotations

from typing import Any

from core.variable_contract import clean_definition
from models.scenario_variable import ScenarioVariableModel
from models.scenario_user_variable import ScenarioUserVariableModel
from models.scenario_proposal import ScenarioProposalModel
from models.selection_lock import SelectionLockModel


def upsert_recommended(requested_scenario_id: str, scenario_version: int, variables: list[dict[str, Any]], actor_user_id: str | None = None) -> int:
    return ScenarioVariableModel.upsert_many(requested_scenario_id, scenario_version, variables, actor_user_id)


def get_recommended(requested_scenario_id: str, scenario_version: int) -> list[dict[str, Any]]:
    return [clean_definition(v) for v in ScenarioVariableModel.get_enabled(requested_scenario_id, scenario_version)]


def set_user_variables(user_id: str, requested_scenario_id: str, scenario_version: int, variables: list[dict[str, Any]], state: str = "SELECTED") -> int:
    return ScenarioUserVariableModel.upsert_many(user_id, requested_scenario_id, scenario_version, variables, state)


def get_user_variables(user_id: str, requested_scenario_id: str, scenario_version: int) -> list[dict[str, Any]]:
    return [clean_definition(v) for v in ScenarioUserVariableModel.get_active(user_id, requested_scenario_id, scenario_version)]


def delete_user_variable(user_id: str, requested_scenario_id: str, scenario_version: int, variable_key: str) -> bool:
    return ScenarioUserVariableModel.delete(user_id, requested_scenario_id, scenario_version, variable_key)


def get_user_variable_records(user_id: str, requested_scenario_id: str, scenario_version: int) -> list[dict[str, Any]]:
    return ScenarioUserVariableModel.get_records(user_id, requested_scenario_id, scenario_version)


def save_proposal(request_id: str, user_id: str | None, requested_scenario_id: str, scenario_version: int, payload: dict[str, Any]) -> None:
    ScenarioProposalModel.save(request_id, user_id, requested_scenario_id, scenario_version, payload)


def get_selection_lock(lock_key: str) -> dict[str, Any] | None:
    return SelectionLockModel.get(lock_key)


def lock_selection(lock_key: str, requested_scenario_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    return SelectionLockModel.put_if_absent(lock_key, requested_scenario_id, payload)
