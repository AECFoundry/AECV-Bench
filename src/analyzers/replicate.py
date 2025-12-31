"""
Replicate API analyzer for floor plan analysis.
Supports models available on Replicate that aren't on OpenRouter.
"""
import json
import os
import re
import time
from typing import Dict, Union, Optional, Any
from ..models.plan_elements import get_json_schema
from .prompts import COUNTING_RULES
from ..parsers.json_parser import extract_json_from_response
from ..utils.image_utils import encode_image_to_base64

try:
    import replicate
    HAS_REPLICATE = True
except ImportError:
    HAS_REPLICATE = False
    replicate = None


class ReplicateAnalyzer:
    """
    Replicate API analyzer with clean separation of concerns.
    Handles Replicate-specific features including model version resolution,
    different model types (DeepSeek VL2/V12 vs others), and streaming responses.
    """
    
    def __init__(self, max_retries: int = 3, timeout: int = 300):
        """
        Initialize the Replicate analyzer.
        
        Args:
            max_retries: Maximum number of retry attempts for failed requests
            timeout: Request timeout in seconds (Replicate vision models are slow)
        """
        self.max_retries = max_retries
        self.timeout = timeout
        self.retry_delay_base = 2

    def _validate_inputs(self, replicate_api_token: Optional[str], json_schema: Optional[Dict]) -> Dict:
        """Validate input parameters and dependencies."""
        if not HAS_REPLICATE:
            raise ImportError(
                "replicate library is required. Install it with: pip install replicate"
            )
        
        return json_schema or get_json_schema()

    def _setup_api_token(self, replicate_api_token: Optional[str]) -> str:
        """Setup and validate API token in environment."""
        # Set API token if provided
        if replicate_api_token:
            os.environ["REPLICATE_API_TOKEN"] = replicate_api_token
        elif "REPLICATE_API_TOKEN" not in os.environ:
            raise ValueError(
                "Replicate API token is required. "
                "Set it via replicate_api_token parameter or REPLICATE_API_TOKEN environment variable."
            )
        
        # Verify API token is set
        api_token = os.environ.get("REPLICATE_API_TOKEN")
        if not api_token:
            raise ValueError("REPLICATE_API_TOKEN environment variable is not set")
            
        return api_token

    def _resolve_model_version(self, model_name: str, api_token: str) -> str:
        """Resolve model name to include version hash if needed."""
        actual_model_name = model_name
        
        # Try to get latest version if model_name doesn't include a version hash
        if ":" not in model_name:
            try:
                parts = model_name.split("/")
                if len(parts) == 2:
                    owner, name = parts
                    client = replicate.Client(api_token=api_token)
                    print(f"[REPLICATE] Fetching latest version for {model_name}...", flush=True)
                    versions = list(client.models.versions.list(owner, name))
                    if versions:
                        latest_version = versions[0].id
                        actual_model_name = f"{owner}/{name}:{latest_version}"
                        print(f"[REPLICATE] Using latest version: {actual_model_name}", flush=True)
                    else:
                        print(f"[REPLICATE] No versions found, using: {model_name}", flush=True)
            except Exception as e:
                # If we can't fetch versions, continue with original model_name
                error_msg = str(e)
                if "404" in error_msg or "not found" in error_msg.lower():
                    print(f"[REPLICATE] Model {model_name} not found. Error: {error_msg}", flush=True)
                else:
                    print(f"[REPLICATE] Could not fetch latest version: {error_msg}, using: {model_name}", flush=True)
        
        return actual_model_name

    def _build_prompt(self, json_schema: Dict) -> str:
        """Build Replicate-specific prompt for floor plan analysis."""
        # Note: DeepSeek VL2 gets confused by full schema definitions, so use a simple example-based prompt
        intro_text = (
            "You are an expert floor-plan analyst trained to interpret architectural "
            "and engineering drawings in any language.\n\n"
            "Your task is to analyze the floor plan image and count the following elements:\n"
            "- Door: TOTAL number of doors (all types)\n"
            "- Window: TOTAL number of windows (all types)\n"
            "- Space: TOTAL count of distinct spaces/rooms (include every enclosed area)\n"
            "- Bedroom: TOTAL number of bedrooms\n"
            "- Toilet: TOTAL number of toilets/WCs\n\n"
            "After counting, return ONLY a JSON object with your counts. Example format:\n"
            '{"Door": 5, "Window": 8, "Space": 10, "Bedroom": 2, "Toilet": 1}\n\n'
            "Return ONLY the JSON object with your actual counts. No explanations, no schema definitions.\n\n"
        )
        return intro_text + COUNTING_RULES

    def _determine_model_type(self, model_name: str) -> Dict[str, bool]:
        """Determine model type for parameter handling."""
        is_deepseek_vl2 = "deepseek-vl2" in model_name.lower()
        is_deepseek_v12_model = "deepseek-v12" in model_name.lower()
        is_deepseek_vision = is_deepseek_vl2 or is_deepseek_v12_model
        is_ocr_model = "ocr" in model_name.lower()
        
        return {
            "is_deepseek_vl2": is_deepseek_vl2,
            "is_deepseek_v12": is_deepseek_v12_model,
            "is_deepseek_vision": is_deepseek_vision,
            "is_ocr_model": is_ocr_model
        }

    def _build_input_params_deepseek(self, image_path: str, prompt_text: str) -> Dict[str, Any]:
        """Build input parameters for DeepSeek models."""
        from pathlib import Path

        # Use Path object - Replicate SDK will handle upload for each call
        abs_image_path = Path(image_path).resolve()

        input_params = {
            "image": abs_image_path,  # Pass Path object for fresh upload each time
            "prompt": prompt_text
        }

        # Debug info
        print(f"[REPLICATE] Image path: {abs_image_path}, size: {abs_image_path.stat().st_size} bytes", flush=True)
        print(f"[REPLICATE] Prompt length: {len(prompt_text)} chars", flush=True)

        return input_params, None  # No file handle to close

    def _build_input_params_generic(self, image_path: str, prompt_text: str, model_info: Dict, 
                                   temperature: float, max_new_tokens: int, 
                                   top_p: float, repetition_penalty: float) -> Dict[str, Any]:
        """Build input parameters for generic models."""
        image_file = open(image_path, "rb")
        
        input_params = {
            "temperature": temperature,
            "text": prompt_text,
            "max_new_tokens": max_new_tokens,
        }
        
        # Try different image parameter names based on model type
        if model_info["is_ocr_model"]:
            input_params["image"] = image_file
        else:
            input_params["image1"] = image_file
        
        # Add optional parameters if they're commonly supported
        if top_p is not None:
            input_params["top_p"] = top_p
        if repetition_penalty is not None:
            input_params["repetition_penalty"] = repetition_penalty
            
        return input_params, image_file

    def _make_request_with_retry(self, model_name: str, input_params: Dict, image_path: str) -> str:
        """Make Replicate API request with retry logic."""
        content = None
        
        for attempt in range(self.max_retries):
            try:
                print(f"[REPLICATE] Processing image: {os.path.basename(image_path)} (attempt {attempt + 1}/{self.max_retries})", flush=True)
                print(f"[REPLICATE] Calling Replicate API (this may take 1-3 minutes for vision models)...", flush=True)
                
                output = replicate.run(model_name, input=input_params)
                content = self._process_streaming_response(output)
                print(f"[REPLICATE] Received response (length: {len(content) if content else 0} chars)", flush=True)
                break
                
            except Exception as e:
                error_msg = str(e)
                is_404 = "404" in error_msg or "not found" in error_msg.lower() or "could not be found" in error_msg.lower()
                
                if is_404 and attempt == 0:
                    self._print_404_guidance(model_name)
                
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay_base ** attempt  # Exponential backoff
                    print(f"[RETRY] Attempt {attempt + 1}/{self.max_retries} failed: {error_msg}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    if is_404:
                        raise Exception(
                            f"Model '{model_name}' not found on Replicate (404). "
                            f"Please verify the model ID at https://replicate.com/explore. "
                            f"Original error: {error_msg}"
                        ) from e
                    else:
                        raise Exception(f"Replicate API call failed after {self.max_retries} attempts: {error_msg}") from e
        
        return content

    def _print_404_guidance(self, model_name: str):
        """Print helpful guidance for 404 errors."""
        print(f"\n[ERROR] Model '{model_name}' not found on Replicate (404).")
        print(f"Please verify the model ID is correct. You can:")
        print(f"  1. Check available models at: https://replicate.com/explore")
        print(f"  2. Search for 'deepseek ocr' on Replicate")
        print(f"  3. Ensure the model ID format is: 'owner/model-name' or 'owner/model-name:version-hash'")
        print(f"  4. Common formats: 'chenxwh/deepseek-ocr' or 'lucataco/deepseek-ocr'")
        print()

    def _process_streaming_response(self, output: Any) -> str:
        """Process Replicate streaming response."""
        if hasattr(output, '__iter__') and not isinstance(output, str):
            # It's an iterator, collect all chunks
            chunks = []
            chunk_count = 0
            for chunk in output:
                chunks.append(str(chunk))
                chunk_count += 1
                if chunk_count % 10 == 0:  # Print progress every 10 chunks
                    print(f"[REPLICATE] Received {chunk_count} chunks...", flush=True)
            return "".join(chunks)
        else:
            # It's already a string or single value
            return str(output)

    def _process_response(self, content: str, image_path: str) -> Union[str, Dict]:
        """Process the response content and extract JSON."""
        # Check if we got content
        if content is None:
            raise ValueError(f"No response received from Replicate API for {image_path}")
        
        # Check for empty content
        if not content or not content.strip():
            raise ValueError(f"Empty response from Replicate API for {image_path}")
        
        # Try to extract JSON from the response
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
            raise ValueError("Empty response from Replicate API")

    def analyze_floorplan(
        self,
        image_path: str,
        model_name: str,
        json_schema: Dict = None,
        replicate_api_token: str = None,
        temperature: float = 0.1,
        max_new_tokens: int = 2048,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        max_retries: int = None,
        url: str = None,  # Ignored - Replicate doesn't use URL parameter
        **kwargs  # Accept any other unexpected kwargs and ignore them
    ) -> Union[str, Dict]:
        """
        Analyze floor plan image using Replicate API.
        
        This is the main method that coordinates all steps of the analysis process.
        """
        # Use provided max_retries or instance default
        retries = max_retries if max_retries is not None else self.max_retries
        
        # Step 1: Validate inputs
        validated_schema = self._validate_inputs(replicate_api_token, json_schema)
        
        # Step 2: Setup API token
        api_token = self._setup_api_token(replicate_api_token)
        
        # Step 3: Resolve model version
        actual_model_name = self._resolve_model_version(model_name, api_token)
        
        # Step 4: Build prompt
        prompt_text = self._build_prompt(validated_schema)
        
        # Step 5: Determine model type and build parameters
        model_info = self._determine_model_type(actual_model_name)
        
        image_file = None
        try:
            if model_info["is_deepseek_vision"]:
                input_params, image_file = self._build_input_params_deepseek(image_path, prompt_text)
            else:
                input_params, image_file = self._build_input_params_generic(
                    image_path, prompt_text, model_info, temperature, 
                    max_new_tokens, top_p, repetition_penalty
                )
            
            # Step 6: Make request with retries
            content = self._make_request_with_retry(actual_model_name, input_params, image_path)
            
            # Step 7: Process response
            return self._process_response(content, image_path)
            
        finally:
            # Ensure file is closed
            if image_file and not image_file.closed:
                image_file.close()


# Create global analyzer instance
_replicate_analyzer = ReplicateAnalyzer()


def analyze_floorplan_replicate(
    image_path: str,
    model_name: str,
    json_schema: Dict = None,
    replicate_api_token: str = None,
    temperature: float = 0.1,
    max_new_tokens: int = 2048,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    max_retries: int = 3,
    url: str = None,  # Ignored - Replicate doesn't use URL parameter
    **kwargs  # Accept any other unexpected kwargs and ignore them
) -> Union[str, Dict]:
    """
    Analyze floor plan using Replicate API.
    
    This function supports models available on Replicate (e.g., DeepSeek VL2).
    Replicate API uses a different format than OpenAI/OpenRouter.
    
    Args:
        image_path: Path to the floor-pla
        n image file
        model_name: The Replicate model identifier (e.g., "chenxwh/deepseek-vl2:8ea887897e772107ce53f3a7fa4850e78ae88b2b73ff854b4700db9f0d59c7cb")
        json_schema: A dict defining the JSON schema (for reference in prompt)
        replicate_api_token: Replicate API token (can also be set via REPLICATE_API_TOKEN env var)
        temperature: Sampling temperature (default 0.1)
        max_new_tokens: Maximum tokens to generate (default 2048)
        top_p: Top-p sampling parameter (default 0.9)
        repetition_penalty: Repetition penalty (default 1.1)
        max_retries: Maximum number of retry attempts
        
    Returns:
        Dictionary with floor plan element counts
        
    Raises:
        ImportError if replicate library is not installed
        ValueError if API token is not provided
        Exception if API request fails
    """
    if not HAS_REPLICATE:
        raise ImportError(
            "replicate library is required. Install it with: pip install replicate"
        )
    
    if json_schema is None:
        json_schema = get_json_schema()
    
    # Set API token if provided
    if replicate_api_token:
        os.environ["REPLICATE_API_TOKEN"] = replicate_api_token
    elif "REPLICATE_API_TOKEN" not in os.environ:
        raise ValueError(
            "Replicate API token is required. "
            "Set it via replicate_api_token parameter or REPLICATE_API_TOKEN environment variable."
        )
    
    # Verify API token is set (Replicate will read from environment)
    api_token = os.environ.get("REPLICATE_API_TOKEN")
    if not api_token:
        raise ValueError("REPLICATE_API_TOKEN environment variable is not set")
    
    # Note: According to Replicate docs, we can use replicate.run() directly
    # It will automatically read the token from REPLICATE_API_TOKEN env var
    # No need to initialize a client explicitly
    
    # Build the prompt - use simple example-based format
    # Note: DeepSeek VL2 gets confused by full schema definitions, so use a clearer approach
    intro_text = (
        "You are an expert floor-plan analyst trained to interpret architectural "
        "and engineering drawings in any language.\n\n"
        "Your task is to analyze the floor plan image and count the following elements:\n"
        "- Door: TOTAL number of doors (all types)\n"
        "- Window: TOTAL number of windows (all types)\n"
        "- Space: TOTAL count of distinct spaces/rooms (include every enclosed area)\n"
        "- Bedroom: TOTAL number of bedrooms\n"
        "- Toilet: TOTAL number of toilets/WCs\n\n"
        "After counting, return ONLY a JSON object with your counts. Example format:\n"
        '{"Door": 5, "Window": 8, "Space": 10, "Bedroom": 2, "Toilet": 1}\n\n'
        "Return ONLY the JSON object with your actual counts. No explanations, no schema definitions.\n\n"
    )
    prompt_text = intro_text + COUNTING_RULES
    
    # Retry logic for network errors
    retry_delay = 2
    content = None
    actual_model_name = model_name  # May be updated if we fetch latest version
    
    # Try to get latest version if model_name doesn't include a version hash
    if ":" not in model_name:
        try:
            parts = model_name.split("/")
            if len(parts) == 2:
                owner, name = parts
                client = replicate.Client(api_token=api_token)
                print(f"[REPLICATE] Fetching latest version for {model_name}...", flush=True)
                versions = list(client.models.versions.list(owner, name))
                if versions:
                    latest_version = versions[0].id
                    actual_model_name = f"{owner}/{name}:{latest_version}"
                    print(f"[REPLICATE] Using latest version: {actual_model_name}", flush=True)
                else:
                    print(f"[REPLICATE] No versions found, using: {model_name}", flush=True)
        except Exception as e:
            # If we can't fetch versions, continue with original model_name
            error_msg = str(e)
            if "404" in error_msg or "not found" in error_msg.lower():
                print(f"[REPLICATE] Model {model_name} not found. Error: {error_msg}", flush=True)
            else:
                print(f"[REPLICATE] Could not fetch latest version: {error_msg}, using: {model_name}", flush=True)
    
    for attempt in range(max_retries):
        try:
            # Replicate API call
            # Note: Replicate API calls can take 30 seconds to several minutes, especially for vision models
            # Using "image1" parameter as per DeepSeek VL2 model schema
            print(f"[REPLICATE] Processing image: {os.path.basename(image_path)} (attempt {attempt + 1}/{max_retries})", flush=True)
            print(f"[REPLICATE] Calling Replicate API (this may take 1-3 minutes for vision models)...", flush=True)
            
            # Check if this is DeepSeek VL2 or V12 (uses different parameter names)
            is_deepseek_vl2 = "deepseek-vl2" in actual_model_name.lower()
            is_deepseek_v12_model = "deepseek-v12" in actual_model_name.lower()
            is_deepseek_vision = is_deepseek_vl2 or is_deepseek_v12_model
            
            # DeepSeek V12/VL2: Use Path object for fresh upload each time
            if is_deepseek_vision:
                from pathlib import Path

                # Use Path object - Replicate SDK will handle upload for each call
                abs_image_path = Path(image_path).resolve()

                input_params = {
                    "image": abs_image_path,  # Pass Path object for fresh upload each time
                    "prompt": prompt_text
                }

                # Debug info
                print(f"[REPLICATE] Image path: {abs_image_path}, size: {abs_image_path.stat().st_size} bytes", flush=True)
                print(f"[REPLICATE] Prompt length: {len(prompt_text)} chars", flush=True)

                output = replicate.run(
                    actual_model_name,
                    input=input_params
                )

                # Replicate returns an iterator for streaming responses
                # Collect all output chunks with progress indication
                if hasattr(output, '__iter__') and not isinstance(output, str):
                    # It's an iterator, collect all chunks
                    chunks = []
                    chunk_count = 0
                    for chunk in output:
                        chunks.append(str(chunk))
                        chunk_count += 1
                        if chunk_count % 10 == 0:  # Print progress every 10 chunks
                            print(f"[REPLICATE] Received {chunk_count} chunks...", flush=True)
                    content = "".join(chunks)
                else:
                    # It's already a string or single value
                    content = str(output)
            else:
                # Other models - open file and pass to Replicate
                # According to Replicate docs, use replicate.run() directly with file object
                # Replicate will handle the file upload automatically
                # Note: Different models may use different parameter names (image1, image, file, etc.)
                with open(image_path, "rb") as image_file:
                    input_params = {
                        "temperature": temperature,
                    }
                    input_params["text"] = prompt_text
                    input_params["max_new_tokens"] = max_new_tokens
                    # Try different image parameter names based on model type
                    # OCR models typically use "image" or "file", vision models use "image1"
                    if "ocr" in actual_model_name.lower():
                        # OCR models typically use "image" parameter
                        input_params["image"] = image_file
                    else:
                        # Vision models typically use "image1" parameter
                        input_params["image1"] = image_file
                    
                    # Add optional parameters if they're commonly supported
                    if top_p is not None:
                        input_params["top_p"] = top_p
                    if repetition_penalty is not None:
                        input_params["repetition_penalty"] = repetition_penalty
                    
                    output = replicate.run(
                        actual_model_name,
                        input=input_params
                    )
                    
                    # Replicate returns an iterator for streaming responses
                    # Collect all output chunks with progress indication
                    if hasattr(output, '__iter__') and not isinstance(output, str):
                        # It's an iterator, collect all chunks
                        chunks = []
                        chunk_count = 0
                        for chunk in output:
                            chunks.append(str(chunk))
                            chunk_count += 1
                            if chunk_count % 10 == 0:  # Print progress every 10 chunks
                                print(f"[REPLICATE] Received {chunk_count} chunks...", flush=True)
                        content = "".join(chunks)
                    else:
                        # It's already a string or single value
                        content = str(output)
            
            print(f"[REPLICATE] Received response (length: {len(content) if content else 0} chars)", flush=True)
            break
            
        except Exception as e:
            error_msg = str(e)
            # Check if it's a 404 error (model not found)
            is_404 = "404" in error_msg or "not found" in error_msg.lower() or "could not be found" in error_msg.lower()
            
            if is_404 and attempt == 0:
                # Provide helpful guidance on first 404 error
                print(f"\n[ERROR] Model '{actual_model_name}' not found on Replicate (404).")
                print(f"Please verify the model ID is correct. You can:")
                print(f"  1. Check available models at: https://replicate.com/explore")
                print(f"  2. Search for 'deepseek ocr' on Replicate")
                print(f"  3. Ensure the model ID format is: 'owner/model-name' or 'owner/model-name:version-hash'")
                print(f"  4. Common formats: 'chenxwh/deepseek-ocr' or 'lucataco/deepseek-ocr'")
                print()
            
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                print(f"[RETRY] Attempt {attempt + 1}/{max_retries} failed: {error_msg}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                if is_404:
                    raise Exception(
                        f"Model '{actual_model_name}' not found on Replicate (404). "
                        f"Please verify the model ID at https://replicate.com/explore. "
                        f"Original error: {error_msg}"
                    ) from e
                else:
                    raise Exception(f"Replicate API call failed after {max_retries} attempts: {error_msg}") from e
    
    # Check if we got content
    if content is None:
        raise ValueError(f"No response received from Replicate API for {image_path}")
    
    # Check for empty content
    if not content or not content.strip():
        raise ValueError(f"Empty response from Replicate API for {image_path}")
    
    # Try to extract JSON from the response - same logic as analyze_floorplan_prompt_based
    import re
    if content:
        # First try direct JSON parsing (same as Claude variant)
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # If direct parsing fails, try to extract JSON from text (same as Claude variant)
            extracted_json = extract_json_from_response(content, image_path)
            if extracted_json:
                return extracted_json
            # If extraction fails, try to find JSON in code blocks (same as Claude variant)
            json_block = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_block:
                try:
                    return json.loads(json_block.group(1))
                except json.JSONDecodeError:
                    pass
            # Last resort: raise error with content preview (same as Claude variant)
            raise ValueError(f"Could not extract valid JSON from response. Content preview: {content[:200]}...")
    else:
        raise ValueError("Empty response from Replicate API")

