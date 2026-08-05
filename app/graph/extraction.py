"""Strict source-only graph extraction with deterministic validation and cache keys."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Protocol

from app.generation.openai_compatible import parse_json_object
from app.graph.models import ChunkGraphExtraction, ExtractedRelation
from app.graph.ontology import validate_relation_types
from app.index_registry.diff import graph_extraction_text


GRAPH_EXTRACTION_SYSTEM_PROMPT = """You extract a closed-ontology scientific graph from one chunk.
Return exactly one JSON object matching the supplied schema. Use only facts explicitly stated
in this chunk; never add model knowledge and never infer across chunks. evidence_quote must be
a contiguous exact substring of the chunk. Co-occurrence is not a causal/functional relation.
Comparing A with B is not evidence that A outperforms B. Separate positive and negative claims
and preserve conditions, scenarios, datasets, satellites, noise values, parameters, formula
symbols, and Chinese/English aliases. Definitions of A and B alone do not establish an A-B
relation. Use only the provided predicates; drop an unclassifiable relation instead of inventing
a predicate. Output strict JSON and no markdown."""


class ChatProvider(Protocol):
    model_name: str
    def chat_completion(self, messages: list[dict[str, str]], *, max_tokens: int | None = None) -> dict[str, Any]: ...


def extraction_cache_key(
    graph_text_hash: str, extractor_model: str, prompt_version: str, ontology_version: str
) -> str:
    return hashlib.sha256("\x1f".join((graph_text_hash, extractor_model,
                                      prompt_version, ontology_version)).encode()).hexdigest()


def validate_extraction(
    extraction: ChunkGraphExtraction, chunk_content: str,
) -> tuple[ChunkGraphExtraction, list[str]]:
    entities = {value.local_id: value for value in extraction.entities}
    valid: list[ExtractedRelation] = []
    issues: list[str] = []
    for index, relation in enumerate(extraction.relations):
        prefix = f"relations[{index}]"
        subject = entities.get(relation.subject_local_id)
        object_value = entities.get(relation.object_local_id)
        if subject is None or object_value is None:
            issues.append(prefix + ": subject/object is missing")
            continue
        if relation.evidence_quote not in chunk_content:
            issues.append(prefix + ": evidence_quote is not a contiguous source substring")
            continue
        allowed, reason = validate_relation_types(
            subject.entity_type, relation.predicate, object_value.entity_type,
            relation.qualifiers,
        )
        if not allowed:
            issues.append(prefix + ": " + reason)
            continue
        try:
            json.dumps(relation.qualifiers, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError):
            issues.append(prefix + ": qualifiers are not JSON serializable")
            continue
        valid.append(relation)
    return extraction.model_copy(update={"relations": valid}), issues


def extract_chunk_graph(
    provider: ChatProvider, chunk: dict[str, Any], *, prompt_version: str = "1.0",
    ontology_version: str = "1.0", max_tokens: int = 4096,
) -> tuple[ChunkGraphExtraction, dict[str, Any]]:
    content = str(chunk.get("content") or "")
    schema = ChunkGraphExtraction.model_json_schema()
    messages = [
        {"role": "system", "content": GRAPH_EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": "Schema:\n" + json.dumps(schema, ensure_ascii=False) +
         "\n\nChunk:\n" + graph_extraction_text(chunk)},
    ]
    payload = provider.chat_completion(messages, max_tokens=max_tokens)
    try:
        raw = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("graph extraction response lacks message.content") from error
    extraction = ChunkGraphExtraction.model_validate(parse_json_object(str(raw)))
    validated, issues = validate_extraction(extraction, content)
    return validated, {"rejected_relation_count": len(issues), "issues": issues,
                       "extractor_model": provider.model_name,
                       "extractor_prompt_version": prompt_version,
                       "ontology_version": ontology_version}
