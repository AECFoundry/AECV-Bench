"""
LLM-based evaluation system for QA benchmark results.

Uses an LLM judge (via OpenRouter) to evaluate if model answers are correct
compared to ground truth. Returns binary scores: 1.0 for correct, 0.0 for incorrect.
"""
import csv
import time
from typing import Dict, Optional
from collections import defaultdict
import numpy as np
import requests

from ..utils.config import get_open_router_api_key, require_api_key


class QAEvaluator:
    """Evaluates QA answers using LLM-as-Judge via OpenRouter."""
    
    def __init__(
        self,
        judge_model: str = "openai/gpt-4o",
        open_router_api_key: Optional[str] = None,
        url: str = "https://openrouter.ai/api/v1/chat/completions",
        temperature: float = 0.0
    ):
        """
        Initialize the QA evaluator.
        
        Args:
            judge_model: Model identifier for the judge LLM (default: openai/gpt-4o)
            open_router_api_key: OpenRouter API key (if None, will try to get from env)
            url: OpenRouter API URL
            temperature: Temperature for judge model (default 0.0 for deterministic)
        """
        self.judge_model = judge_model
        self.url = url
        self.temperature = temperature
        
        # Get API key
        if open_router_api_key is None or not open_router_api_key.strip():
            self.api_key = require_api_key("OPEN_ROUTER_API_KEY", "OpenRouter")
        else:
            self.api_key = open_router_api_key.strip()
    
    def judge_answer(self, question: str, ground_truth: str, model_answer: str) -> float:
        """
        Use LLM to judge if the model answer is correct compared to ground truth.
        
        Args:
            question: The question asked
            ground_truth: The correct answer
            model_answer: The model's answer
            
        Returns:
            1.0 if the answer is correct/contextually similar, 0.0 if incorrect/opposite
        """
        # Prepare the evaluation prompt
        prompt = f"""You are an expert evaluator for question-answering tasks. Your task is to determine if a model's answer is correct compared to the ground truth answer.

Question: {question}

Ground Truth Answer: {ground_truth}

Model Answer: {model_answer}

Evaluate whether the model's answer is correct and contextually similar to the ground truth. Consider:
- If the model answer correctly addresses the question
- If the key information matches the ground truth (even if phrased differently)
- If the answer is factually correct and not contradictory

Respond with ONLY a single number:
- 1 if the answer is correct/contextually similar
- 0 if the answer is incorrect, contradictory, or completely different

Your response (just the number, nothing else):"""

        # Prepare headers
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Build the message payload
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        payload = {
            "model": self.judge_model,
            "messages": messages,
            "temperature": self.temperature
        }

        # Make API call with retry logic
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                resp = requests.post(self.url, headers=headers, json=payload, timeout=60)
                resp.raise_for_status()
                break
            except requests.exceptions.HTTPError as e:
                error_msg = f"{e}"
                try:
                    if hasattr(e, 'response') and e.response is not None:
                        error_detail = e.response.json() if e.response.content else {}
                        error_msg = f"{e}: {error_detail}"
                except:
                    pass
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"[RETRY] Attempt {attempt + 1}/{max_retries} failed: {error_msg}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"[ERROR] Failed to evaluate after {max_retries} attempts: {error_msg}")
                    return 0.0
            except (requests.exceptions.ConnectionError, 
                    requests.exceptions.Timeout,
                    requests.exceptions.RequestException) as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"[RETRY] Attempt {attempt + 1}/{max_retries} failed: {type(e).__name__}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"[ERROR] Failed to evaluate after {max_retries} attempts: {e}")
                    return 0.0

        # Parse response
        try:
            resp_json = resp.json()
            content = resp_json["choices"][0]["message"]["content"].strip()
            
            # Extract score (look for 1 or 0 in the response)
            if not content:
                print(f"[WARN] Empty response from judge model")
                return 0.0
            
            # Try to extract 1 or 0 from response
            content_lower = content.lower()
            if '1' in content or 'correct' in content_lower or 'yes' in content_lower:
                # Check if it's explicitly 0 or incorrect
                if '0' in content or 'incorrect' in content_lower or 'wrong' in content_lower or 'no' in content_lower:
                    # If both present, check which comes first or is more explicit
                    if content.strip() == '0' or content.strip().startswith('0'):
                        return 0.0
                    elif content.strip() == '1' or content.strip().startswith('1'):
                        return 1.0
                    # Default to checking if incorrect keywords are present
                    if any(word in content_lower for word in ['incorrect', 'wrong', 'no', 'different']):
                        return 0.0
                return 1.0
            elif '0' in content or 'incorrect' in content_lower or 'wrong' in content_lower:
                return 0.0
            else:
                # Default: try to parse as float/int
                try:
                    score = float(content.strip())
                    return 1.0 if score >= 0.5 else 0.0
                except:
                    print(f"[WARN] Could not parse judge response: {content}. Defaulting to 0.0")
                    return 0.0
                    
        except (KeyError, IndexError, ValueError) as e:
            print(f"[ERROR] Failed to parse judge response: {e}")
            return 0.0
    
    def evaluate_answer(self, question: str, ground_truth: str, model_answer: str, task: str = "", qa_type: str = "") -> Dict[str, float]:
        """
        Evaluate a single answer using LLM judge.
        
        Args:
            question: The question asked
            ground_truth: Ground truth answer
            model_answer: Model's answer
            task: Task type (optional, for statistics)
            qa_type: QA type (optional, for statistics)
            
        Returns:
            Dictionary with evaluation score (binary: 1.0 or 0.0)
        """
        # Skip error answers
        if model_answer.startswith('[ERROR:'):
            score = 0.0
        else:
            score = self.judge_answer(question, ground_truth, model_answer)
        
        return {
            'score': score,
            'overall': score  # For compatibility with existing code
        }
    
    def evaluate_csv(self, csv_path: str) -> Dict:
        """
        Evaluate all answers in a QA results CSV file.
        
        Args:
            csv_path: Path to the CSV file with QA results
            
        Returns:
            Dictionary with evaluation results and statistics
        """
        results = []
        task_stats = defaultdict(lambda: {'total': 0, 'scores': []})
        qa_type_stats = defaultdict(lambda: {'total': 0, 'scores': []})
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            total_rows = len(rows)
            
            print(f"Evaluating {total_rows} question-answer pairs...")
            
            for idx, row in enumerate(rows, 1):
                question = row.get('question', '').strip()
                predicted = row.get('model_answer', '').strip()
                ground_truth = row.get('ground_truth', '').strip()
                task = row.get('task', 'unknown')
                qa_type = row.get('qa_type', 'unknown')
                qa_id = row.get('qa_id', '')
                image_id = row.get('image_id', '')
                
                # Evaluate using LLM judge
                if idx % 10 == 0:
                    print(f"  Progress: {idx}/{total_rows} ({100*idx/total_rows:.1f}%)")
                
                scores = self.evaluate_answer(question, ground_truth, predicted, task, qa_type)
                
                result = {
                    'image_id': image_id,
                    'qa_id': qa_id,
                    'task': task,
                    'qa_type': qa_type,
                    'question': question,
                    'ground_truth': ground_truth,
                    'predicted': predicted,
                    'score': scores['score'],
                    'overall': scores['overall']
                }
                results.append(result)
                
                # Update statistics
                task_stats[task]['total'] += 1
                task_stats[task]['scores'].append(scores['overall'])
                
                qa_type_stats[qa_type]['total'] += 1
                qa_type_stats[qa_type]['scores'].append(scores['overall'])
        
        # Compute aggregate statistics
        if results:
            overall_scores = [r['overall'] for r in results]
            summary = {
                'total_questions': len(results),
                'mean_overall_score': np.mean(overall_scores),
                'std_overall_score': np.std(overall_scores),
                'task_breakdown': {
                    task: {
                        'total': stats['total'],
                        'mean_score': np.mean(stats['scores']) if stats['scores'] else 0.0,
                        'std_score': np.std(stats['scores']) if stats['scores'] else 0.0
                    }
                    for task, stats in task_stats.items()
                },
                'qa_type_breakdown': {
                    qa_type: {
                        'total': stats['total'],
                        'mean_score': np.mean(stats['scores']) if stats['scores'] else 0.0,
                        'std_score': np.std(stats['scores']) if stats['scores'] else 0.0
                    }
                    for qa_type, stats in qa_type_stats.items()
                }
            }
        else:
            summary = {'total_questions': 0}
        
        return {
            'results': results,
            'summary': summary
        }
