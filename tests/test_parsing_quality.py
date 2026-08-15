from __future__ import annotations

from app.parsing.quality import validate_canonical_document


def test_quality_validator_removes_control_characters_and_records_issue() -> None:
    document = {
        "blocks": [
            {
                "block_id": "P_test_p001_b001",
                "type": "paragraph",
                "text": "A valid sentence with a bad\x15 control character.",
                "quality": {
                    "status": "usable",
                    "issues": [],
                    "retrieval_enabled": True,
                },
            }
        ]
    }

    report = validate_canonical_document(document)

    block = document["blocks"][0]
    assert "\x15" not in block["text"]
    assert "control_character" in block["quality"]["issues"]
    assert block["quality"]["retrieval_enabled"] is True
    assert report["sanitized_block_count"] == 1
    assert report["issue_counts"] == {"control_character": 1}


def test_quality_validator_rejects_malformed_formula_and_empty_table() -> None:
    document = {
        "blocks": [
            {
                "block_id": "P_test_p001_b001",
                "type": "equation",
                "latex": r"x = \frac{1}{2",
                "quality": {
                    "status": "usable",
                    "issues": [],
                    "retrieval_enabled": True,
                },
            },
            {
                "block_id": "P_test_p001_b002",
                "type": "table",
                "caption": "",
                "text": "",
                "image_path": None,
                "table_html": None,
                "quality": {
                    "status": "usable",
                    "issues": [],
                    "retrieval_enabled": True,
                },
            },
        ]
    }

    report = validate_canonical_document(document)

    equation, table = document["blocks"]
    assert "formula_unbalanced_delimiter" in equation["quality"]["issues"]
    assert equation["quality"]["retrieval_enabled"] is False
    assert "empty_table" in table["quality"]["issues"]
    assert table["quality"]["retrieval_enabled"] is False
    assert report["invalid_block_count"] == 2


def test_quality_validator_does_not_guess_spelling_corrections() -> None:
    document = {
        "blocks": [
            {
                "block_id": "P_test_p001_b001",
                "type": "paragraph",
                "text": "The diferent models sufer from a source-level typo.",
                "quality": {
                    "status": "usable",
                    "issues": [],
                    "retrieval_enabled": True,
                },
            }
        ]
    }

    validate_canonical_document(document)

    assert document["blocks"][0]["text"] == (
        "The diferent models sufer from a source-level typo."
    )
