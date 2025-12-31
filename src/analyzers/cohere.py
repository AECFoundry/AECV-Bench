"""
Cohere API analyzer for floor plan analysis.
"""
import os
import time
import requests
from typing import Dict, Union, Optional
from ..utils.image_utils import encode_image_to_base64
from ..parsers.json_parser import extract_json_counts
from ..parsers.text_parser import parse_counts_from_text_improved


class CohereAnalyzer:
    """
    Cohere API analyzer with clean separation of concerns.
    Handles Cohere v2 API specifics including response parsing and refusal detection.
    """
    
    def __init__(self, max_retries: int = 3, timeout: int = 60):
        """
        Initialize the Cohere analyzer.
        
        Args:
            max_retries: Maximum number of retry attempts for failed requests
            timeout: Request timeout in seconds
        """
        self.max_retries = max_retries
        self.timeout = timeout
        self.retry_delay_base = 2
        self.max_tokens = 2000

    def _validate_inputs(self, cohere_api_key: Optional[str], json_schema: Dict) -> str:
        """Validate input parameters."""
        if not cohere_api_key:
            raise ValueError("Cohere API key is required")
        return cohere_api_key.strip()

    def _prepare_headers(self, cohere_api_key: str) -> Dict[str, str]:
        """Prepare HTTP headers for Cohere API."""
        return {
            'accept': 'application/json',
            'content-type': 'application/json',
            'Authorization': f'bearer {cohere_api_key}'
        }

    def _build_prompt(self, image_name: str) -> str:
        """Build Cohere-specific prompt for floor plan analysis."""
        return (
            f"FLOOR PLAN ANALYSIS\n"
            f"Image: {image_name}\n\n"
            "I have attached a floor plan image for you to analyze. Please examine the image carefully and count the following elements:\n\n"
            "1. DOORS - Count all door openings, gaps in walls, entrance points (both interior and exterior doors)\n"
            "2. WINDOWS - Count all window openings shown on exterior walls (rectangular symbols on walls)\n"
            "3. SPACES - Count every distinct room or area (bedrooms, living room, kitchen, bathrooms, closets, hallways, any enclosed space)\n"
            "4. BEDROOMS - Count only rooms that are clearly bedrooms (sleeping areas)\n"
            "5. TOILETS - Count bathrooms, WCs, toilet rooms\n\n"
            "The image is attached below. Please analyze it and provide your counts.\n\n"
            "IMPORTANT: The image is already provided in this message. Please analyze it and give me actual numbers based on what you observe.\n\n"
            "End your response with this exact format:\n"
            'FINAL COUNTS: {"Door": X, "Window": Y, "Space": Z, "Bedroom": A, "Toilet": B}\n\n'
            "Replace X, Y, Z, A, B with the actual numbers you counted from the image."
        )

    def _build_payload(self, model_name: str, prompt_text: str, data_url: str, temperature: float) -> Dict:
        """Build the request payload for Cohere v2 API."""
        return {
            "model": model_name,
            "messages": [
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
            ],
            "temperature": temperature,
            "max_tokens": self.max_tokens
        }

    def _make_request_with_retry(self, url: str, headers: Dict, payload: Dict, image_path: str) -> Dict:
        """Make HTTP request with retry logic for network errors."""
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                return resp.json()
            except requests.HTTPError as e:
                error_msg = f"{e}"
                try:
                    if hasattr(e, 'response') and e.response is not None:
                        error_detail = e.response.json() if e.response.content else {}
                        error_msg = f"{e}: {error_detail}"
                except:
                    pass
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay_base ** attempt  # Exponential backoff
                    print(f"[RETRY] Attempt {attempt + 1}/{self.max_retries} failed: {error_msg}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise requests.exceptions.HTTPError(f"{error_msg} for {image_path}") from e
            except (requests.exceptions.ConnectionError, 
                    requests.exceptions.Timeout,
                    requests.exceptions.RequestException) as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay_base ** attempt  # Exponential backoff
                    print(f"[RETRY] Attempt {attempt + 1}/{self.max_retries} failed: {type(e).__name__}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise

    def _extract_content_from_response(self, resp_json: Dict) -> str:
        """Extract content from Cohere v2 API response structure."""
        try:
            # Try different v2 response structures
            if 'message' in resp_json:
                if 'content' in resp_json['message']:
                    if isinstance(resp_json['message']['content'], list):
                        return resp_json['message']['content'][0].get('text', '')
                    else:
                        return resp_json['message']['content']
            elif 'text' in resp_json:
                return resp_json['text']
            elif 'choices' in resp_json:
                return resp_json['choices'][0]['message']['content']
            else:
                return str(resp_json)
        except (KeyError, IndexError, TypeError):
            return str(resp_json)

    def _check_model_refusal(self, content: str) -> bool:
        """Check if the model refused to analyze the image."""
        refusal_indicators = [
            "cannot analyze",
            "can't analyze",
            "unable to process",
            "don't have the ability",
            "cannot see",
            "can't see",
            "i can't view",
            "i'm sorry, but i can't"
        ]
        return any(indicator in content.lower() for indicator in refusal_indicators)

    def _process_response(self, content: str, image_name: str) -> Dict:
        """Process the response content and extract counts."""
        # Try to extract JSON from response
        json_result = extract_json_counts(content, image_name)
        if json_result:
            return json_result

        # Fallback to text parsing
        text_result = parse_counts_from_text_improved(content, image_name)
        return text_result

    def analyze_floorplan(
        self,
        image_path: str,
        model_name: str,
        json_schema: Dict,
        cohere_api_key: str,
        url: str = "https://api.cohere.com/v2/chat",
        temperature: float = 0.0,
        max_retries: int = None,
    ) -> Dict:
        """
        Analyze floor plan using Cohere v2 API.
        
        Args:
            image_path: Path to the floor-plan image file
            model_name: The model identifier to use
            json_schema: A dict defining the JSON schema (for reference in prompt)
            cohere_api_key: Cohere API key
            url: The endpoint URL (default is Cohere v2 chat)
            temperature: Sampling temperature (default 0.0 for deterministic)
            max_retries: Override default max_retries for this request
            
        Returns:
            Dictionary with floor plan element counts
        """
        # Use provided max_retries or instance default
        retries = max_retries if max_retries is not None else self.max_retries
        
        # Step 1: Validate inputs
        validated_api_key = self._validate_inputs(cohere_api_key, json_schema)
        
        # Step 2: Prepare image
        base64_image = encode_image_to_base64(image_path)
        data_url = f"data:image/png;base64,{base64_image}"
        img_name = os.path.basename(image_path)
        
        # Step 3: Prepare headers
        headers = self._prepare_headers(validated_api_key)
        
        # Step 4: Build prompt
        prompt_text = self._build_prompt(img_name)
        
        # Step 5: Build payload
        payload = self._build_payload(model_name, prompt_text, data_url, temperature)

        # Step 6: Retry logic for handling model inconsistencies
        for attempt in range(retries):
            # Make request with retries
            resp_json = self._make_request_with_retry(url, headers, payload, image_path)
            
            # Extract content from response
            content = self._extract_content_from_response(resp_json)
            
            # Check if model refused to analyze
            if self._check_model_refusal(content):
                if attempt < retries - 1:
                    continue  # Retry
                else:
                    return {"Door": 0, "Window": 0, "Space": 0, "Bedroom": 0, "Toilet": 0, "error": "Model refusal"}

            # Process response and extract counts
            result = self._process_response(content, img_name)
            
            # Check if we got meaningful results (not all zeros)
            if sum(result.values()) > 0 or attempt == retries - 1:
                return result
            else:
                continue
        
        # Final fallback
        return {"Door": 0, "Window": 0, "Space": 0, "Bedroom": 0, "Toilet": 0, "error": "Max retries exceeded"}


# Create global analyzer instance
_cohere_analyzer = CohereAnalyzer()


def analyze_floorplan_cohere(
    image_path: str,
    model_name: str,
    json_schema: Dict,
    cohere_api_key: str,
    url: str = "https://api.cohere.com/v2/chat",
    temperature: float = 0.0,
    max_retries: int = 3,
) -> Dict:
    """
    Analyze floor plan using Cohere v2 API with better debugging and response handling.
    
    Args:
        image_path: Path to the floor-plan image file
        model_name: The model identifier to use
        json_schema: A dict defining the JSON schema (for reference in prompt)
        cohere_api_key: Cohere API key
        url: The endpoint URL (default is Cohere v2 chat)
        temperature: Sampling temperature (default 0.0 for deterministic)
        max_retries: Maximum number of retry attempts
        
    Returns:
        Dictionary with floor plan element counts
    """
    return _cohere_analyzer.analyze_floorplan(
        image_path=image_path,
        model_name=model_name,
        json_schema=json_schema,
        cohere_api_key=cohere_api_key,
        url=url,
        temperature=temperature,
        max_retries=max_retries
    )

