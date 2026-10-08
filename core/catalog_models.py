"""Request and response contracts of the scenario catalog API (the choices behind the UI dropdowns)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DataType = Literal["transactional", "aggregational"]
_ID = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---- import: the same nested shape the bundled catalog uses ------------------------------------------------------
class ScenarioIn(_Strict):
    scenarioId: str = Field(pattern=_ID, description="The requestedScenarioId used by /scenario/propose, for example LB-01")
    scenarioType: str = Field(min_length=1, max_length=100)
    businessScenario: str | None = Field(default=None, max_length=12000)
    active: bool = True


class JourneyIn(_Strict):
    id: str | None = Field(default=None, pattern=_ID, description="Optional; derived from the names above it when omitted")
    name: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=1000)
    dataTypes: list[DataType] = Field(default_factory=lambda: ["transactional"], min_length=1)
    industryType: str | None = Field(default=None, max_length=100, description="industryType sent to /scenario/propose; the industry's name when omitted")
    domain: str | None = Field(default=None, max_length=200, description="domain sent to /scenario/propose (the pair its source documents are uploaded under); the journey's name when omitted")
    entityKey: str | None = Field(default=None, max_length=200)
    active: bool = True
    scenarios: list[ScenarioIn] = Field(default_factory=list)


class UseCaseIn(_Strict):
    id: str | None = Field(default=None, pattern=_ID)
    name: str = Field(min_length=1, max_length=200)
    proposeValue: str | None = Field(default=None, max_length=200, description="useCase sent to /scenario/propose; the name when omitted")
    description: str | None = Field(default=None, max_length=1000)
    active: bool = True
    journeys: list[JourneyIn] = Field(default_factory=list)


class SubDomainIn(_Strict):
    id: str | None = Field(default=None, pattern=_ID)
    name: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=1000)
    active: bool = True
    useCases: list[UseCaseIn] = Field(default_factory=list)


class IndustryIn(_Strict):
    id: str | None = Field(default=None, pattern=_ID)
    name: str = Field(min_length=1, max_length=100)
    description: str | None = Field(default=None, max_length=1000)
    active: bool = True
    subDomains: list[SubDomainIn] = Field(default_factory=list)


class CatalogImportRequest(_Strict):
    industries: list[IndustryIn] = Field(min_length=1)


class LevelCount(BaseModel):
    created: int = 0
    updated: int = 0


class CatalogImportResponse(BaseModel):
    success: bool = True
    industries: LevelCount
    subDomains: LevelCount
    useCases: LevelCount
    journeys: LevelCount
    scenarios: LevelCount


# ---- reads -------------------------------------------------------------------------------------------------------
class CatalogItem(BaseModel):
    id: str
    name: str
    description: str | None = None
    order: int = 0


class CatalogJourney(CatalogItem):
    description: str | None = None
    dataTypes: list[str]
    industryType: str
    domain: str
    entityKey: str | None = None


class CatalogScenario(BaseModel):
    scenarioId: str
    scenarioType: str
    label: str = Field(description="What the dropdown shows, for example 'LB-01: Normal'")
    order: int = 0
    businessScenario: str | None = None


class CatalogScenarioList(BaseModel):
    journeyId: str
    count: int
    allLabel: str = Field(description="The 'all scenarios' option, for example 'All 9 Scenarios for Selected Journey'")
    scenarios: list[CatalogScenario]


class CatalogProposeDefaults(BaseModel):
    """What /scenario/propose needs that the catalog knows; the client adds country and anything it lets the user edit."""
    requestedScenarioId: str
    scenarioType: str
    industryType: str
    domain: str
    useCase: str
    typeOfData: str
    businessScenario: str | None = None
    entityKey: str | None = None


class CatalogScenarioDetail(CatalogScenario):
    industry: CatalogItem
    subDomain: CatalogItem
    useCase: CatalogItem
    journey: CatalogJourney
    proposeRequest: CatalogProposeDefaults


class CatalogScenarioNode(CatalogScenario):
    pass


class CatalogJourneyNode(CatalogJourney):
    scenarios: list[CatalogScenarioNode]


class CatalogUseCaseNode(CatalogItem):
    journeys: list[CatalogJourneyNode]


class CatalogSubDomainNode(CatalogItem):
    useCases: list[CatalogUseCaseNode]


class CatalogIndustryNode(CatalogItem):
    subDomains: list[CatalogSubDomainNode]


class CatalogTree(BaseModel):
    industries: list[CatalogIndustryNode]


class CatalogCountry(BaseModel):
    code: str = Field(description="Value for the country field of /scenario/propose")
    name: str
    currency: str | None = None
