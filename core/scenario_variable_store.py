"""Application service for scenario variable persistence.

MongoDB collection access lives in the models package; this module keeps the
business-facing service API used by the workflow and HTTP layer.
"""
from __future__ import annotations

from typing import Any

from models.scenario_variable import ScenarioVariableModel
from models.scenario_user_variable import ScenarioUserVariableModel
from models.scenario_proposal import ScenarioProposalModel


def upsert_recommended(scenario_key: str, scenario_version: int, variables: list[dict[str, Any]], actor_user_id: str | None = None) -> int:
    return ScenarioVariableModel.upsert_many(scenario_key, scenario_version, variables, actor_user_id)


def get_recommended(scenario_key: str, scenario_version: int) -> list[dict[str, Any]]:
    return ScenarioVariableModel.get_enabled(scenario_key, scenario_version)


def set_user_variables(user_id: str, scenario_key: str, scenario_version: int, variables: list[dict[str, Any]], state: str = "SELECTED") -> int:
    return ScenarioUserVariableModel.upsert_many(user_id, scenario_key, scenario_version, variables, state)


def get_user_variables(user_id: str, scenario_key: str, scenario_version: int) -> list[dict[str, Any]]:
    return ScenarioUserVariableModel.get_active(user_id, scenario_key, scenario_version)


def delete_user_variable(user_id: str, scenario_key: str, scenario_version: int, variable_key: str) -> bool:
    return ScenarioUserVariableModel.delete(user_id, scenario_key, scenario_version, variable_key)


def get_user_variable_records(user_id: str, scenario_key: str, scenario_version: int) -> list[dict[str, Any]]:
    return ScenarioUserVariableModel.get_records(user_id, scenario_key, scenario_version)


def save_proposal(request_id: str, user_id: str | None, scenario_key: str, scenario_version: int, payload: dict[str, Any]) -> None:
    ScenarioProposalModel.save(request_id, user_id, scenario_key, scenario_version, payload)
