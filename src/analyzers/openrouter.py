"""
OpenRouter API analyzer for floor plan analysis.
"""
import json
import re
import requests
import time
from typing import Dict, Union, Optional
from ..utils.image_utils import encode_image_to_base64
from ..models.plan_elements import get_json_schema
from .prompts import COUNTING_RULES
from ..parsers.json_parser import extract_json_from_response

try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    HAS_TENACITY = True
except ImportError:
    HAS_TENACITY = False


class OpenRouterAnalyzer:
    """
    Configurable OpenRouter API analyzer that eliminates code duplication.
    Supports both standard JSON schema enforcement and Claude-style prompt-based extraction.
    """
    
    def __init__(self, use_schema_format: bool = True):
        """
        Initialize the analyzer.
        
        Args:
            use_schema_format: If True, uses JSON schema enforcement in API request.
                             If False, uses prompt-based JSON extraction (Claude-style).
        """
        self.use_schema_format = use_schema_format
        self.max_retries = 3
        self.retry_delay = 2
        self.timeout = 60

    def _validate_inputs(self, image_path: str, model_name: str, open_router_api_key: Optional[str], json_schema: Optional[Dict]):
        """Validate input parameters."""
        if open_router_api_key is None:
            raise ValueError("OpenRouter API key is required")
        
        return json_schema or get_json_schema()

    def _prepare_headers(self, open_router_api_key: str) -> Dict[str, str]:
        """Prepare HTTP headers for the request."""
        return {
            "Authorization": f"Bearer {open_router_api_key}",
            "Content-Type": "application/json"
        }

    def _build_prompt(self, json_schema: Dict) -> str:
        """Build the prompt text based on configuration."""
        if self.use_schema_format:
            # Standard format - schema enforcement via API
            intro_text = (
                "You are an expert floor-plan analyst trained to interpret architectural "
                "and engineering drawings in any language.\n\n"
                "Your task is to fully understand the drawing first, then count the elements exactly "
                "according to the rules below, and then return ONLY the JSON strictly following the provided schema."
            )
        else:
            # Claude format - schema in prompt
            schema_str = json.dumps(json_schema, indent=2)
            intro_text = (
                f"You are an expert floor-plan analyst trained to interpret architectural "
                f"and engineering drawings in any language.\n\n"
                f"Your task is to fully understand the drawing first, then count the elements exactly "
                f"according to the rules below, and then return ONLY the JSON strictly following the provided schema:\n\n"
                f"{schema_str}\n\n"
            )
        
        return intro_text + "\n\n" + COUNTING_RULES

    def _build_payload(self, model_name: str, prompt_text: str, data_url: str, temperature: float, json_schema: Dict) -> Dict:
        """Build the request payload based on configuration."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_text
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url
                        }
                    }
                ]
            }
        ]

        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature
        }

        # Add schema enforcement for standard format
        if self.use_schema_format:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "PlanElements",
                    "strict": True,
                    "schema": json_schema,
                }
            }

        return payload

    def _make_request_with_retry(self, url: str, headers: Dict, payload: Dict, image_path: str) -> Dict:
        """Make HTTP request with retry logic."""
        for attempt in range(self.max_retries):
            try:
                print(
                    f"[REQUEST] Sending to OpenRouter (attempt {attempt + 1}/{self.max_retries}, "
                    f"timeout={self.timeout}s) for '{image_path}' with model '{payload.get('model')}'",
                    flush=True,
                )
                resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.HTTPError as e:
                # For HTTP errors, try to get more details from the response
                error_msg = f"{e}"
                try:
                    if hasattr(e, 'response') and e.response is not None:
                        error_detail = e.response.json() if e.response.content else {}
                        error_msg = f"{e}: {error_detail}"
                except:
                    pass
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    print(f"[RETRY] Attempt {attempt + 1}/{self.max_retries} failed: {error_msg}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise requests.exceptions.HTTPError(f"{error_msg} for {image_path}") from e
            except (requests.exceptions.ConnectionError, 
                    requests.exceptions.Timeout,
                    requests.exceptions.RequestException) as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    print(f"[RETRY] Attempt {attempt + 1}/{self.max_retries} failed: {type(e).__name__}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise

    def _extract_content_from_response(self, resp_json: Dict, image_path: str) -> str:
        """Extract content from API response."""
        try:
            content = resp_json["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise KeyError(f"Unexpected response format: {resp_json}") from e
        
        # Check for empty content
        if not content or (self.use_schema_format and not content.strip()):
            raise ValueError(f"Empty response from API for {image_path}. Response: {resp_json}")
        
        return content

    def _process_response(self, content: str, image_path: str) -> Union[str, Dict]:
        """Process the response content based on configuration."""
        if self.use_schema_format:
            # Standard format - return content directly (should be valid JSON)
            return content
        else:
            # Claude format - extract JSON from potentially mixed content
            if content:
                # First try direct JSON parsing
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    # If direct parsing fails, try to extract JSON from text
                    extracted_json = extract_json_from_response(content, image_path)
                    if extracted_json:
                        return extracted_json
                    # If extraction fails, try to find JSON in code blocks
                    json_block = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
                    if json_block:
                        try:
                            return json.loads(json_block.group(1))
                        except json.JSONDecodeError:
                            pass
                    # Last resort: raise error with content preview
                    raise ValueError(f"Could not extract valid JSON from response. Content preview: {content[:200]}...")
            else:
                raise ValueError("Empty response from API")

    def analyze_floorplan(
        self,
        image_path: str,
        model_name: str,
        json_schema: Dict = None,
        open_router_api_key: str = None,
        url: str = "https://openrouter.ai/api/v1/chat/completions",
        temperature: float = 0.0,
    ) -> Union[str, Dict]:
        """
        Analyze floor plan image using OpenRouter API.
        
        This is the main method that coordinates all steps of the analysis process.
        """
        # Step 1: Validate inputs
        validated_schema = self._validate_inputs(image_path, model_name, open_router_api_key, json_schema)
        
        # Step 2: Prepare image
        base64_image = encode_image_to_base64(image_path)
        data_url = f"data:image/png;base64,{base64_image}"
        
        # Step 3: Prepare headers
        headers = self._prepare_headers(open_router_api_key)
        
        # Step 4: Build prompt
        prompt_text = self._build_prompt(validated_schema)
        
        # Step 5: Build payload
        payload = self._build_payload(model_name, prompt_text, data_url, temperature, validated_schema)
        
        # Step 6: Make request with retries
        resp_json = self._make_request_with_retry(url, headers, payload, image_path)
        
        # Step 7: Extract content
        content = self._extract_content_from_response(resp_json, image_path)
        
        # Step 8: Process response
        return self._process_response(content, image_path)


# Create global analyzer instances
_standard_analyzer = OpenRouterAnalyzer(use_schema_format=True)
_prompt_based_analyzer = OpenRouterAnalyzer(use_schema_format=False)


def analyze_floorplan(
    image_path: str,
    model_name: str,
    json_schema: Dict = None,
    open_router_api_key: str = None,
    url: str = "https://openrouter.ai/api/v1/chat/completions",
    temperature: float = 0.0,
) -> Union[str, Dict]:
    """
    Sends a floor-plan image to the OpenRouter chat-completions endpoint and returns the JSON response.

    Parameters:
    - image_path: Path to the floor-plan image file.
    - model_name: The model identifier to use (e.g., "google/gemini-2.0-flash-001").
    - json_schema: A dict defining the JSON schema expected in the response.
    - open_router_api_key: Your OpenRouter API key (Bearer token).
    - url: The endpoint URL (default is OpenRouter chat-completions).
    - temperature: Sampling temperature (default 0.0 for deterministic).

    Returns:
    - The parsed JSON content from the API response (the "content" field under choices[0].message).

    Raises:
    - requests.HTTPError if the response status is not 2xx.
    - KeyError if the expected fields are missing in the JSON response.
    """
    return _standard_analyzer.analyze_floorplan(
        image_path=image_path,
        model_name=model_name,
        json_schema=json_schema,
        open_router_api_key=open_router_api_key,
        url=url,
        temperature=temperature
    )


def analyze_floorplan_prompt_based(
    image_path: str,
    model_name: str,
    json_schema: Dict = None,
    open_router_api_key: str = None,
    url: str = "https://openrouter.ai/api/v1/chat/completions",
    temperature: float = 0.0,
) -> Union[str, Dict]:
    """
    Sends a floor-plan image to the OpenRouter chat-completions endpoint.
    Uses prompt-based JSON extraction instead of strict schema enforcement.

    Use this for models that don't support structured output via response_format.

    Parameters:
    - image_path: Path to the floor-plan image file.
    - model_name: The model identifier to use.
    - json_schema: A dict defining the JSON schema expected in the response.
    - open_router_api_key: Your OpenRouter API key (Bearer token).
    - url: The endpoint URL (default is OpenRouter chat-completions).
    - temperature: Sampling temperature (default 0.0 for deterministic).

    Returns:
    - The parsed JSON content from the API response.

    Raises:
    - requests.HTTPError if the response status is not 2xx.
    - KeyError if the expected fields are missing in the JSON response.
    """
    return _prompt_based_analyzer.analyze_floorplan(
        image_path=image_path,
        model_name=model_name,
        json_schema=json_schema,
        open_router_api_key=open_router_api_key,
        url=url,
        temperature=temperature
    )

