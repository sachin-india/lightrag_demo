"""
LLM-as-Judge metrics for evaluating RAG responses using GPT-4

Uses Alloy LLM handler to evaluate responses on multiple dimensions:
- Correctness: Factual accuracy of the answer
- Completeness: Coverage of the question requirements  
- Faithfulness: Grounding in provided context (no hallucinations)
- Conciseness: Appropriate detail level without excessive verbosity

Integrates with Intel's Alloy platform for LLM access.
"""

import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

# Add parent directory to path for alloy imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import Alloy LLM handler (cannot be modified, use as-is)
from alloy.llm import chat
from alloy.llm.workflow import WorkflowSession, generate_workflow_session_id

logger = logging.getLogger(__name__)


class LLMJudge:
    """
    LLM-as-Judge evaluator using GPT-4 via Alloy platform
    
    Evaluates RAG responses on multiple quality dimensions using structured prompts.
    Uses async wrappers around Alloy's synchronous chat function for pipeline compatibility.
    """
    
    def __init__(self, 
                 model: str = "openai-azure-gpt4o",
                 temperature: float = 0.0,
                 max_concurrent: int = 8):
        """
        Initialize LLM Judge
        
        Args:
            model: Alloy model to use (default: GPT-4o for best judgment quality)
            temperature: Temperature for generation (0.0 for consistency)
            max_concurrent: Maximum concurrent LLM calls (increased from 3 to 8 for faster evaluation)
        """
        self.model = model
        self.temperature = temperature
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # Workflow tracking for all judge evaluations
        self.workflow_id = "benchmark_llm_judge"
        
        logger.info(f"✅ LLM Judge initialized: {model}")
    
    def _build_judge_prompt(self, 
                           query: str,
                           reference_answer: str,
                           predicted_answer: str,
                           context: Optional[str] = None) -> str:
        """
        Build structured prompt for LLM judge evaluation
        
        Args:
            query: Original question
            reference_answer: Ground truth answer
            predicted_answer: System's predicted answer
            context: Optional retrieved context for faithfulness check
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""You are an expert evaluator assessing the quality of AI-generated answers to questions.

**Question:**
{query}

**Reference Answer (Ground Truth):**
{reference_answer}

**Predicted Answer (System Output):**
{predicted_answer}
"""
        
        if context:
            prompt += f"""
**Retrieved Context:**
{context[:500]}... (truncated)
"""
        
        prompt += """
**Task:**
Evaluate the Predicted Answer on the following dimensions using a 1-5 scale:

1. **Correctness** (1-5): Does the answer contain factually accurate information?
   - 1: Completely incorrect or contradicts facts
   - 3: Partially correct with some errors
   - 5: Fully correct and accurate

2. **Completeness** (1-5): Does the answer fully address all aspects of the question?
   - 1: Missing critical information
   - 3: Covers main points but lacks some details
   - 5: Comprehensively answers all aspects

3. **Faithfulness** (1-5): Is the answer grounded in the provided context without hallucinations?
   - 1: Contains significant unsupported claims
   - 3: Mostly grounded with minor extrapolations
   - 5: Fully grounded in provided information

4. **Conciseness** (1-5): Is the answer appropriately detailed without excessive verbosity?
   - 1: Extremely verbose or overly brief
   - 3: Reasonable balance with some unnecessary details
   - 5: Perfect balance of detail and brevity

**Output Format:**
Return ONLY a valid JSON object with this exact structure:
```json
{
  "correctness": <score 1-5>,
  "completeness": <score 1-5>,
  "faithfulness": <score 1-5>,
  "conciseness": <score 1-5>,
  "explanation": "<brief 1-2 sentence justification>"
}
```

Respond with ONLY the JSON object, no additional text.
"""
        
        return prompt
    
    async def _call_alloy_async(self, prompt: str, session_id: str) -> str:
        """
        Async wrapper around Alloy's synchronous chat function
        
        Args:
            prompt: Prompt to send
            session_id: Workflow session ID for tracking
            
        Returns:
            Response string from Alloy LLM
        """
        loop = asyncio.get_event_loop()
        
        # Run sync chat function in thread pool executor to make it async
        response = await loop.run_in_executor(
            None,
            lambda: chat(
                prompt=prompt,
                model=self.model,
                type="azure",
                temperature=self.temperature,
                max_retries=3,
                workflow_id=self.workflow_id,
                workflow_session_id=session_id
            )
        )
        
        return response
    
    def _parse_judge_response(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON response from LLM judge
        
        Args:
            response: Raw response from LLM
            
        Returns:
            Dictionary with scores and explanation
        """
        # Check for Alloy error responses
        if isinstance(response, str) and response.startswith("Error:"):
            logger.error(f"Alloy returned error: {response}")
            return {
                'correctness': 0,
                'completeness': 0,
                'faithfulness': 0,
                'conciseness': 0,
                'explanation': f"Judge error: {response}",
                'error': True
            }
        
        try:
            # Extract JSON from response (may have markdown code fences)
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON object directly
                json_match = re.search(r'\{.*?\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = response
            
            # Parse JSON
            result = json.loads(json_str)
            
            # Validate required fields
            required_fields = ['correctness', 'completeness', 'faithfulness', 'conciseness']
            for field in required_fields:
                if field not in result:
                    logger.warning(f"Missing field '{field}' in judge response")
                    result[field] = 0
            
            # Add error flag
            result['error'] = False
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse judge response as JSON: {e}")
            logger.error(f"Response: {response}")
            return {
                'correctness': 0,
                'completeness': 0,
                'faithfulness': 0,
                'conciseness': 0,
                'explanation': 'Failed to parse judge response',
                'error': True
            }
    
    async def judge_response(self,
                           query: str,
                           reference_answer: str,
                           predicted_answer: str,
                           context: Optional[str] = None,
                           session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate a single response using LLM judge
        
        Args:
            query: Original question
            reference_answer: Ground truth answer
            predicted_answer: System's predicted answer
            context: Optional retrieved context
            session_id: Optional workflow session ID (generated if not provided)
            
        Returns:
            Dictionary with scores for each dimension
        """
        # Use semaphore to limit concurrent LLM calls
        async with self.semaphore:
            # Generate session ID if not provided
            if session_id is None:
                session_id = generate_workflow_session_id()
            
            # Build prompt
            prompt = self._build_judge_prompt(
                query=query,
                reference_answer=reference_answer,
                predicted_answer=predicted_answer,
                context=context
            )
            
            # Call Alloy LLM asynchronously
            response = await self._call_alloy_async(prompt, session_id)
            
            # Parse and return scores
            scores = self._parse_judge_response(response)
            
            return scores
    
    async def judge_batch(self,
                         evaluations: List[Dict[str, str]],
                         session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Evaluate multiple responses concurrently
        
        Args:
            evaluations: List of dicts with 'query', 'reference_answer', 
                        'predicted_answer', and optional 'context'
            session_id: Optional workflow session ID for grouping
            
        Returns:
            List of score dictionaries
        """
        # Generate session ID for this batch
        if session_id is None:
            session_id = generate_workflow_session_id()
        
        logger.info(f"🤖 Evaluating {len(evaluations)} responses with LLM judge...")
        
        # Create tasks for concurrent evaluation
        tasks = [
            self.judge_response(
                query=eval_item['query'],
                reference_answer=eval_item['reference_answer'],
                predicted_answer=eval_item['predicted_answer'],
                context=eval_item.get('context'),
                session_id=session_id
            )
            for eval_item in evaluations
        ]
        
        # Execute concurrently with semaphore limiting parallelism
        results = await asyncio.gather(*tasks)
        
        logger.info(f"✅ Completed {len(results)} judge evaluations")
        
        return results
    
    def calculate_all(self,
                     query: str,
                     reference_answer: str,
                     predicted_answer: str,
                     context: Optional[str] = None) -> Dict[str, Any]:
        """
        Synchronous wrapper for single evaluation (for compatibility)
        
        Args:
            query: Original question
            reference_answer: Ground truth answer
            predicted_answer: System's predicted answer
            context: Optional retrieved context
            
        Returns:
            Dictionary with scores for each dimension
        """
        # Run async function in new event loop (for sync callers)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                self.judge_response(query, reference_answer, predicted_answer, context)
            )
            return result
        finally:
            loop.close()


# Testing and example usage
if __name__ == "__main__":
    async def test_llm_judge():
        print("🧪 Testing LLM Judge with Alloy")
        print("="*70)
        
        judge = LLMJudge(max_concurrent=2)
        
        # Test case 1: Good answer
        test_cases = [
            {
                'query': 'What is the capital of France?',
                'reference_answer': 'Paris',
                'predicted_answer': 'The capital of France is Paris, which is also the largest city in the country.',
                'context': 'Paris is the capital and most populous city of France.'
            },
            {
                'query': 'What is RBA?',
                'reference_answer': 'Results-Based Accountability',
                'predicted_answer': 'RBA stands for Results-Based Accountability, which is a disciplined way of thinking and taking action that communities can use to improve outcomes.',
                'context': 'Results-Based Accountability (RBA) is a framework for improving program and community results.'
            }
        ]
        
        print("\n📝 Test Cases:")
        for i, case in enumerate(test_cases, 1):
            print(f"\n{i}. Query: {case['query']}")
            print(f"   Reference: {case['reference_answer']}")
            print(f"   Predicted: {case['predicted_answer'][:100]}...")
        
        print("\n🤖 Calling LLM Judge...")
        results = await judge.judge_batch(test_cases)
        
        print("\n📊 Judge Scores:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {test_cases[i-1]['query']}")
            if result.get('error'):
                print(f"   ❌ Error: {result['explanation']}")
            else:
                print(f"   Correctness:  {result['correctness']}/5")
                print(f"   Completeness: {result['completeness']}/5")
                print(f"   Faithfulness: {result['faithfulness']}/5")
                print(f"   Conciseness:  {result['conciseness']}/5")
                print(f"   Explanation: {result.get('explanation', 'N/A')}")
        
        print("\n" + "="*70)
        print("✅ LLM Judge test complete!")
    
    # Run test
    asyncio.run(test_llm_judge())
