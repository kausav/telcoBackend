"""Initialize MongoDB model indexes in one explicit place."""
from models.scenario import ScenarioModel
from models.scenario_draft import ScenarioDraftModel
from models.scenario_variable import ScenarioVariableModel
from models.scenario_user_variable import ScenarioUserVariableModel
from models.industry_source import IndustrySourceModel
from models.selection_lock import SelectionLockModel
from models.generation_spec import GenerationSpecModel
from models.catalog import CATALOG_MODELS

ALL_MODELS = (
    ScenarioModel,
    ScenarioDraftModel,
    ScenarioVariableModel,
    ScenarioUserVariableModel,
    IndustrySourceModel,
    SelectionLockModel,
    GenerationSpecModel,
    *CATALOG_MODELS,
)


def ensure_indexes() -> None:
    for model in ALL_MODELS:
        model.ensure_indexes()
