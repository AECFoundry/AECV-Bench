"""
JSON extraction utilities for parsing model responses.
"""
import json
import re
from typing import Dict, Optional


def extract_json_counts(content: str, img_name: str = "") -> Optional[Dict]:
    """
    Extract JSON counts from response with multiple patterns.
    
    Args:
        content: The response content string
        img_name: Optional image name for logging
        
    Returns:
        Dictionary with counts or None if extraction fails
    """
    # Look for FINAL COUNTS pattern first
    final_counts_match = re.search(r'FINAL COUNTS:\s*(\{[^}]*\})', content, re.IGNORECASE)
    if final_counts_match:
        try:
            result = json.loads(final_counts_match.group(1))
            return result
        except json.JSONDecodeError:
            pass

    # Look for any complete JSON with all fields
    complete_json_pattern = r'\{[^{}]*"Door"[^{}]*"Window"[^{}]*"Space"[^{}]*"Bedroom"[^{}]*"Toilet"[^{}]*\}'
    json_match = re.search(complete_json_pattern, content, re.IGNORECASE)
    if json_match:
        try:
            result = json.loads(json_match.group())
            return result
        except json.JSONDecodeError:
            pass

    # Look for any JSON-like structure
    json_patterns = [
        r'\{[^{}]*"Door"[^{}]*\}',
        r'\{"Door":\s*\d+[^}]*\}',
    ]

    for pattern in json_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        for match in matches:
            try:
                result = json.loads(match)
                if isinstance(result, dict) and 'Door' in result:
                    # Fill missing fields with 0
                    for field in ["Door", "Window", "Space", "Bedroom", "Toilet"]:
                        if field not in result:
                            result[field] = 0
                    return result
            except json.JSONDecodeError:
                continue

    return None


def extract_json_from_response(content: str, img_name: str = "") -> Optional[Dict]:
    """
    Extract JSON from various response formats.
    
    Args:
        content: The response content string
        img_name: Optional image name for logging
        
    Returns:
        Dictionary with counts or None if extraction fails
    """
    # Try different JSON extraction patterns
    patterns = [
        r'\{[^{}]*"Door"[^{}]*"Window"[^{}]*"Space"[^{}]*"Bedroom"[^{}]*"Toilet"[^{}]*\}',  # Complete JSON
        r'\{"Door":\s*\d+,\s*"Window":\s*\d+,\s*"Space":\s*\d+,\s*"Bedroom":\s*\d+,\s*"Toilet":\s*\d+\}',  # Exact format
        r'\{[^{}]*"Door"[^{}]*\}',  # Any JSON with Door field
    ]

    for pattern in patterns:
        matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
        for match in matches:
            try:
                result = json.loads(match)
                if isinstance(result, dict) and 'Door' in result:
                    return result
            except json.JSONDecodeError:
                continue

    return None

