"""Initialize MongoDB model indexes in one explicit place."""
from models.scenario import ScenarioModel
from models.scenario_draft import ScenarioDraftModel
from models.scenario_feedback import ScenarioFeedbackModel
from models.scenario_variable import ScenarioVariableModel
from models.scenario_user_variable import ScenarioUserVariableModel
from models.scenario_proposal import ScenarioProposalModel
from models.conversation import ConversationModel
from models.chat_message import ChatMessageModel
from models.registry_entity import RegistryEntityModel
from models.registry_standard import RegistryStandardModel
from models.registry_meta import RegistryMetaModel

ALL_MODELS = (
    ScenarioModel,
    ScenarioDraftModel,
    ScenarioFeedbackModel,
    ScenarioVariableModel,
    ScenarioUserVariableModel,
    ScenarioProposalModel,
    ConversationModel,
    ChatMessageModel,
    RegistryEntityModel,
    RegistryStandardModel,
    RegistryMetaModel,
)


def ensure_indexes() -> None:
    for model in ALL_MODELS:
        model.ensure_indexes()
