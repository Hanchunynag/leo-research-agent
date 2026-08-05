"""Closed, type-aware LEO scientific relation ontology."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class EntityType(StrEnum):
    MEASUREMENT = "measurement"
    OBSERVABLE = "observable"
    SIGNAL = "signal"
    SATELLITE = "satellite"
    CONSTELLATION = "constellation"
    STATE = "state"
    PARAMETER = "parameter"
    PRIOR = "prior"
    INPUT = "input"
    METHOD = "method"
    MODEL = "model"
    ALGORITHM = "algorithm"
    DATASET = "dataset"
    METRIC = "metric"
    RESULT = "result"
    ASSUMPTION = "assumption"
    CONDITION = "condition"
    SCENARIO = "scenario"
    ERROR_SOURCE = "error_source"
    COORDINATE_FRAME = "coordinate_frame"
    PAPER = "paper"
    ORGANIZATION = "organization"
    OTHER = "other"


class RelationPredicate(StrEnum):
    USES = "USES"
    OBSERVES = "OBSERVES"
    CONSTRAINS = "CONSTRAINS"
    ESTIMATES = "ESTIMATES"
    INPUT_TO = "INPUT_TO"
    PRIOR_FOR = "PRIOR_FOR"
    OUTPUT_OF = "OUTPUT_OF"
    IMPROVES = "IMPROVES"
    DEGRADES = "DEGRADES"
    AFFECTS = "AFFECTS"
    EVALUATED_ON = "EVALUATED_ON"
    MEASURED_BY = "MEASURED_BY"
    COMPARED_WITH = "COMPARED_WITH"
    OUTPERFORMS = "OUTPERFORMS"
    UNDERPERFORMS = "UNDERPERFORMS"
    PART_OF = "PART_OF"
    DERIVED_FROM = "DERIVED_FROM"
    APPLIES_TO = "APPLIES_TO"
    ASSUMES = "ASSUMES"
    REQUIRES = "REQUIRES"
    PRODUCES = "PRODUCES"
    CORRELATED_WITH = "CORRELATED_WITH"
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    EQUIVALENT_TO = "EQUIVALENT_TO"
    DIFFERS_FROM = "DIFFERS_FROM"


@dataclass(frozen=True)
class RelationRule:
    subject_types: frozenset[EntityType]
    object_types: frozenset[EntityType]
    directed: bool = True
    reverse_inference: bool = False
    multi_hop: bool = True
    qualifier_required: bool = False
    aggregate: bool = True


ALL = frozenset(EntityType)
METHODS = frozenset({EntityType.METHOD, EntityType.MODEL, EntityType.ALGORITHM})
OBSERVATIONS = frozenset({EntityType.MEASUREMENT, EntityType.OBSERVABLE, EntityType.SIGNAL})
STATES = frozenset({EntityType.STATE, EntityType.PARAMETER, EntityType.ERROR_SOURCE})
EVALUATION = frozenset({EntityType.DATASET, EntityType.SCENARIO, EntityType.CONDITION})


RULES: dict[RelationPredicate, RelationRule] = {
    RelationPredicate.USES: RelationRule(METHODS, ALL),
    RelationPredicate.OBSERVES: RelationRule(METHODS | frozenset({EntityType.SATELLITE}), OBSERVATIONS | STATES),
    RelationPredicate.CONSTRAINS: RelationRule(OBSERVATIONS | frozenset({EntityType.PRIOR}), STATES),
    RelationPredicate.ESTIMATES: RelationRule(METHODS, STATES),
    RelationPredicate.INPUT_TO: RelationRule(OBSERVATIONS | frozenset({EntityType.INPUT, EntityType.PRIOR, EntityType.DATASET}), METHODS),
    RelationPredicate.PRIOR_FOR: RelationRule(frozenset({EntityType.PRIOR, EntityType.MODEL, EntityType.PARAMETER}), METHODS),
    RelationPredicate.OUTPUT_OF: RelationRule(frozenset({EntityType.RESULT, EntityType.STATE, EntityType.PARAMETER}), METHODS),
    RelationPredicate.IMPROVES: RelationRule(ALL, frozenset({EntityType.RESULT, EntityType.METRIC, EntityType.STATE}), qualifier_required=False),
    RelationPredicate.DEGRADES: RelationRule(ALL, frozenset({EntityType.RESULT, EntityType.METRIC, EntityType.STATE}), qualifier_required=False),
    RelationPredicate.AFFECTS: RelationRule(ALL, ALL),
    RelationPredicate.EVALUATED_ON: RelationRule(METHODS, EVALUATION | frozenset({EntityType.DATASET})),
    RelationPredicate.MEASURED_BY: RelationRule(frozenset({EntityType.RESULT, EntityType.STATE, EntityType.ERROR_SOURCE}), frozenset({EntityType.METRIC, EntityType.MEASUREMENT})),
    RelationPredicate.COMPARED_WITH: RelationRule(ALL, ALL, directed=False, reverse_inference=True),
    RelationPredicate.OUTPERFORMS: RelationRule(METHODS | frozenset({EntityType.RESULT}), METHODS | frozenset({EntityType.RESULT}), qualifier_required=True),
    RelationPredicate.UNDERPERFORMS: RelationRule(METHODS | frozenset({EntityType.RESULT}), METHODS | frozenset({EntityType.RESULT}), qualifier_required=True),
    RelationPredicate.PART_OF: RelationRule(ALL, ALL),
    RelationPredicate.DERIVED_FROM: RelationRule(ALL, ALL),
    RelationPredicate.APPLIES_TO: RelationRule(METHODS | frozenset({EntityType.RESULT}), EVALUATION | frozenset({EntityType.SATELLITE, EntityType.CONSTELLATION, EntityType.SIGNAL})),
    RelationPredicate.ASSUMES: RelationRule(METHODS | frozenset({EntityType.MODEL}), frozenset({EntityType.ASSUMPTION, EntityType.CONDITION})),
    RelationPredicate.REQUIRES: RelationRule(ALL, ALL),
    RelationPredicate.PRODUCES: RelationRule(METHODS | frozenset({EntityType.SIGNAL, EntityType.SATELLITE}), frozenset({EntityType.RESULT, EntityType.MEASUREMENT, EntityType.OBSERVABLE, EntityType.STATE, EntityType.PARAMETER})),
    RelationPredicate.CORRELATED_WITH: RelationRule(ALL, ALL, directed=False, reverse_inference=True),
    RelationPredicate.SUPPORTS: RelationRule(frozenset({EntityType.RESULT, EntityType.PAPER, EntityType.METHOD}), ALL),
    RelationPredicate.CONTRADICTS: RelationRule(ALL, ALL, directed=False, reverse_inference=True),
    RelationPredicate.EQUIVALENT_TO: RelationRule(ALL, ALL, directed=False, reverse_inference=True),
    RelationPredicate.DIFFERS_FROM: RelationRule(ALL, ALL, directed=False, reverse_inference=True),
}


def validate_relation_types(
    subject_type: EntityType | str, predicate: RelationPredicate | str,
    object_type: EntityType | str, qualifiers: dict[str, object] | None = None,
) -> tuple[bool, str]:
    try:
        subject = EntityType(subject_type)
        relation = RelationPredicate(predicate)
        object_value = EntityType(object_type)
    except ValueError as error:
        return False, f"unknown ontology value: {error}"
    rule = RULES[relation]
    if subject not in rule.subject_types:
        return False, f"{subject.value} cannot be subject of {relation.value}"
    if object_value not in rule.object_types:
        return False, f"{object_value.value} cannot be object of {relation.value}"
    if rule.qualifier_required and not qualifiers:
        return False, f"{relation.value} requires at least one qualifier"
    return True, ""
