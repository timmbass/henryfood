"""Mapping utilities for food enrichment (starter)
"""

import logging

logger = logging.getLogger(__name__)


def map_to_off(item_name: str) -> dict:
    """Stub: map diary item to OFF entry."""
    logger.info("Would map '%s' to OFF", item_name)
    return {"name": item_name}


def assign_nova_score(item: dict) -> int:
    logger.info("Would assign NOVA score for %s", item.get("name"))
    return 1


def map_ingredients(text: str):
    print(f"Stub: map_ingredients called for {text}")
    return []
