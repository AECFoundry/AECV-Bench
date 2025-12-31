"""
Validation utilities for floor plan results.
"""
from typing import Dict


def validate_floor_plan_result(result: Dict) -> bool:
    """
    Validate that the result looks reasonable.
    
    Args:
        result: Dictionary with floor plan element counts
        
    Returns:
        True if result is valid, False otherwise
    """
    if not isinstance(result, dict):
        return False

    required_fields = ["Door", "Window", "Space", "Bedroom", "Toilet"]

    for field in required_fields:
        if field not in result:
            return False

        value = result[field]
        if not isinstance(value, (int, float)) or value < 0 or value > 1000:
            return False

    # Basic sanity checks
    if result["Bedroom"] > result["Space"]:  # Can't have more bedrooms than total spaces
        return False

    if result["Toilet"] > result["Space"]:  # Can't have more toilets than total spaces
        return False

    return True

