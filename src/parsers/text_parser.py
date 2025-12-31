"""
Text parsing utilities for extracting counts from unstructured responses.
"""
import re
from typing import Dict


def parse_counts_from_text_improved(content: str, img_name: str = "") -> Dict:
    """
    Improved text parsing with more comprehensive patterns.
    
    Args:
        content: The response content string
        img_name: Optional image name for logging
        
    Returns:
        Dictionary with extracted counts (defaults to 0 if not found)
    """
    # More comprehensive patterns
    patterns = {
        "Door": [
            r'(\d+)\s+doors?',
            r'doors?[:\s-]*(\d+)',
            r'door.*?count.*?(\d+)',
            r'(\d+).*?door',
            r'entrance.*?(\d+)',
            r'opening.*?(\d+)'
        ],
        "Window": [
            r'(\d+)\s+windows?',
            r'windows?[:\s-]*(\d+)',
            r'window.*?count.*?(\d+)',
            r'(\d+).*?window'
        ],
        "Space": [
            r'(\d+)\s+(?:spaces?|rooms?|areas?)',
            r'(?:spaces?|rooms?|areas?)[:\s-]*(\d+)',
            r'(?:space|room|area).*?count.*?(\d+)',
            r'(\d+).*?(?:space|room|area)',
            r'total.*?(?:space|room).*?(\d+)'
        ],
        "Bedroom": [
            r'(\d+)\s+bedrooms?',
            r'bedrooms?[:\s-]*(\d+)',
            r'bedroom.*?count.*?(\d+)',
            r'(\d+).*?bedroom',
            r'sleeping.*?(\d+)'
        ],
        "Toilet": [
            r'(\d+)\s+(?:toilets?|bathrooms?|wcs?)',
            r'(?:toilets?|bathrooms?|wcs?)[:\s-]*(\d+)',
            r'(?:toilet|bathroom|wc).*?count.*?(\d+)',
            r'(\d+).*?(?:toilet|bathroom|wc)'
        ]
    }

    result = {}
    content_lower = content.lower()

    for field, field_patterns in patterns.items():
        found_value = None
        found_pattern = None

        for pattern in field_patterns:
            matches = re.findall(pattern, content_lower)
            if matches:
                try:
                    found_value = int(matches[0])
                    found_pattern = pattern
                    break
                except (ValueError, IndexError):
                    continue

        if found_value is not None:
            result[field] = found_value
        else:
            result[field] = 0

    return result


def parse_counts_from_text(text: str) -> Dict:
    """
    Parse numbers from text when JSON extraction fails.
    
    Args:
        text: The response text
        
    Returns:
        Dictionary with extracted counts (defaults to 0 if not found)
    """
    # Patterns to find numbers associated with each element type
    patterns = {
        "Door": [
            r'(\d+)\s+doors?',
            r'doors?[:\s]+(\d+)',
            r'door.*?(\d+)',
            r'(\d+).*?door'
        ],
        "Window": [
            r'(\d+)\s+windows?',
            r'windows?[:\s]+(\d+)',
            r'window.*?(\d+)',
            r'(\d+).*?window'
        ],
        "Space": [
            r'(\d+)\s+(?:spaces?|rooms?)',
            r'(?:spaces?|rooms?)[:\s]+(\d+)',
            r'(?:space|room).*?(\d+)',
            r'(\d+).*?(?:space|room)'
        ],
        "Bedroom": [
            r'(\d+)\s+bedrooms?',
            r'bedrooms?[:\s]+(\d+)',
            r'bedroom.*?(\d+)',
            r'(\d+).*?bedroom'
        ],
        "Toilet": [
            r'(\d+)\s+(?:toilets?|bathrooms?)',
            r'(?:toilets?|bathrooms?)[:\s]+(\d+)',
            r'(?:toilet|bathroom).*?(\d+)',
            r'(\d+).*?(?:toilet|bathroom)'
        ]
    }

    result = {}
    text_lower = text.lower()

    for field, field_patterns in patterns.items():
        found_value = None

        for pattern in field_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                try:
                    # Take the first valid number found
                    found_value = int(matches[0])
                    break
                except (ValueError, IndexError):
                    continue

        result[field] = found_value if found_value is not None else 0

    return result

