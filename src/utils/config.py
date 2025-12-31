"""
Configuration management for API keys and settings.
"""
import os
from typing import Optional
from pathlib import Path


def load_dotenv_if_available():
    """Load .env file if python-dotenv is available and .env exists."""
    try:
        from dotenv import load_dotenv
        env_file = Path('.env')
        if env_file.exists():
            load_dotenv(env_file)
    except ImportError:
        pass


def get_api_key(env_var_name: str) -> Optional[str]:
    """
    Get API key from environment variable.
    
    Args:
        env_var_name: Environment variable name (e.g., 'OPEN_ROUTER_API_KEY')
    
    Returns:
        API key string or None if not set
    """
    # Load .env file each time to pick up any changes
    load_dotenv_if_available()
    return os.getenv(env_var_name)


def require_api_key(env_var_name: str, provider_name: str = None) -> str:
    """
    Require an API key to be set, raise error if not.
    
    Args:
        env_var_name: Environment variable name
        provider_name: Human-readable provider name for error message
        
    Returns:
        The API key string
        
    Raises:
        ValueError: If key is None or empty
    """
    key = get_api_key(env_var_name)
    if not key or not key.strip():
        provider_display = provider_name or env_var_name
        raise ValueError(
            f"{provider_display} API key is required. "
            f"Please set {env_var_name} as an environment variable or create a .env file."
        )
    return key.strip()


# Legacy functions for backward compatibility
def get_open_router_api_key() -> Optional[str]:
    return get_api_key('OPEN_ROUTER_API_KEY')


def get_cohere_api_key() -> Optional[str]:
    return get_api_key('COHERE_API_KEY')

