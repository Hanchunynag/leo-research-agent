from app.indexing.tokenization import tokenize, tokenize_bm25


def test_hyphenated_terms_keep_compound_and_component_tokens() -> None:
    tokens = tokenize_bm25("carrier-phase error")

    assert "carrier-phase" in tokens
    assert "carrier" in tokens
    assert "phase" in tokens


def test_generic_token_count_does_not_include_bm25_aliases() -> None:
    assert tokenize("carrier-phase") == ["carrier-phase"]
