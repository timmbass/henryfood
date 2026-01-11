"""Stub module for src.features.fuzzy_match used by tests."""

__all__ = ["best_match"]

def best_match(query, choices):
    return choices[0] if choices else None
