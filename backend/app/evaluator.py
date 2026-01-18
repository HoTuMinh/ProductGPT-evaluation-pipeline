"""
Core LLM Judge Evaluator using Google Gemini or Groq
"""
import google.generativeai as genai
from typing import Dict, List, Optional, Tuple
import asyncio
import time
import json
from datetime import datetime
import logging
import os

# Try to import groq
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("Warning: Groq not installed. Install with: pip install groq")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMJudge:
    """LLM-as-a-Judge evaluator using Gemini or Groq"""
    
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash-exp",
        temperature: float = 0.2,
        max_tokens: int = 2048,
        provider: str = "gemini"
    ):
        """
        Initialize LLM Judge
        
        Args:
            api_key: API key (Gemini or Groq)
            model: Model name to use
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            provider: "gemini" or "groq"
        """
        self.provider = provider.lower()
        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if self.provider == "groq":
            if not GROQ_AVAILABLE:
                raise ImportError("Groq not installed. Install with: pip install groq")
            self.client = Groq(api_key=api_key)
        else:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model)
            # Generation config
            self.generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
    
    def _build_accuracy_prompt(
        self,
        question: str,
        response: str,
        benchmark: str
    ) -> str:
        """Build prompt for accuracy evaluation"""
        return f"""You are an expert evaluator assessing the ACCURACY of a chatbot response.

**Task**: Compare the chatbot's response to the benchmark answer and evaluate how accurate it is.

**Question**: 
{question}

**Chatbot Response**:
{response}

**Benchmark Answer** (Ground Truth):
{benchmark}

**Evaluation Criteria**:
- Does the response provide factually correct information compared to the benchmark?
- Are the key facts, numbers, and details accurate?
- Are there any factual errors or contradictions?

**Instructions**:
1. Carefully compare the response with the benchmark answer
2. Assign a score from 0.0 to 1.0:
   - 1.0: Perfectly accurate, all facts match
   - 0.7-0.9: Mostly accurate with minor discrepancies
   - 0.4-0.6: Partially accurate, some errors
   - 0.1-0.3: Largely inaccurate
   - 0.0: Completely inaccurate

3. Provide a label: "positive" (score >= 0.7) or "negative" (score < 0.7)
4. Give detailed reasoning explaining your evaluation

**Output Format** (JSON only, no markdown):
{{
    "score": 0.85,
    "label": "positive",
    "reasoning": "The response accurately provides... However, there is a minor discrepancy..."
}}"""

    def _build_comprehensiveness_prompt(
        self,
        question: str,
        response: str,
        benchmark: str
    ) -> str:
        """Build prompt for comprehensiveness evaluation"""
        return f"""You are an expert evaluator assessing the COMPREHENSIVENESS of a chatbot response.

**Task**: Evaluate how complete and thorough the chatbot's response is compared to the benchmark.

**Question**: 
{question}

**Chatbot Response**:
{response}

**Benchmark Answer** (Expected Complete Answer):
{benchmark}

**Evaluation Criteria**:
- Does the response cover all important points from the benchmark?
- Is the level of detail sufficient?
- Are there any missing critical information or aspects?
- Does it provide additional helpful context when appropriate?

**Instructions**:
1. Identify key points that should be covered from the benchmark
2. Check which points are present in the response
3. Assign a score from 0.0 to 1.0:
   - 1.0: Fully comprehensive, covers all key points with appropriate detail
   - 0.7-0.9: Mostly comprehensive, covers most points
   - 0.4-0.6: Partially comprehensive, missing some important points
   - 0.1-0.3: Incomplete, many missing points
   - 0.0: Not comprehensive at all

4. Provide a label: "positive" (score >= 0.6) or "negative" (score < 0.6)
5. Give detailed reasoning explaining what is covered and what is missing

**Output Format** (JSON only, no markdown):
{{
    "score": 0.75,
    "label": "positive",
    "reasoning": "The response covers key points A, B, and C comprehensively. However, it lacks detail on..."
}}"""

    def _build_faithfulness_prompt(
        self,
        question: str,
        response: str,
        benchmark: str
    ) -> str:
        """Build prompt for faithfulness evaluation"""
        return f"""You are an expert evaluator assessing the FAITHFULNESS of a chatbot response.

**Task**: Evaluate whether the chatbot's response is faithful to the source material (benchmark) and doesn't hallucinate or add unsupported information.

**Question**: 
{question}

**Chatbot Response**:
{response}

**Source Material** (Benchmark - What can be verified):
{benchmark}

**Evaluation Criteria**:
- Is all information in the response supported by the source material?
- Does the response avoid making up facts or adding unsupported claims?
- Does it stay within the boundaries of what can be verified from the source?
- Are citations or references (if any) accurate?

**Instructions**:
1. Check each claim in the response against the benchmark
2. Identify any hallucinated or unsupported information
3. Assign a score from 0.0 to 1.0:
   - 1.0: Perfectly faithful, all claims are supported
   - 0.7-0.9: Mostly faithful, very minor unsupported details
   - 0.4-0.6: Partially faithful, some unsupported claims
   - 0.1-0.3: Largely unfaithful, many unsupported claims
   - 0.0: Completely unfaithful, mostly hallucinated

4. Provide a label: "positive" (score >= 0.7) or "negative" (score < 0.7)
5. Give detailed reasoning explaining which parts are faithful and which are not

**Output Format** (JSON only, no markdown):
{{
    "score": 0.90,
    "label": "positive",
    "reasoning": "The response is faithful to the source. All claims about X, Y are verifiable. The only minor issue is..."
}}"""

    def _parse_json_response(self, text: str) -> Dict:
        """Parse JSON from LLM response, handling markdown code blocks"""
        # Remove markdown code blocks if present
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {text[:200]}...")
            # Return default structure if parsing fails
            return {
                "score": 0.0,
                "label": "negative",
                "reasoning": f"Error parsing LLM response: {str(e)}"
            }
    
    async def evaluate_single(
        self,
        metric: str,
        question: str,
        response: str,
        benchmark: str
    ) -> Dict:
        """
        Evaluate a single response for a specific metric
        
        Args:
            metric: One of 'accuracy', 'comprehensiveness', 'faithfulness'
            question: The user's question
            response: The chatbot's response to evaluate
            benchmark: The benchmark/reference answer
            
        Returns:
            Dict with score, label, reasoning, and metadata
        """
        start_time = time.time()
        
        # Build appropriate prompt based on metric
        if metric == "accuracy":
            prompt = self._build_accuracy_prompt(question, response, benchmark)
        elif metric == "comprehensiveness":
            prompt = self._build_comprehensiveness_prompt(question, response, benchmark)
        elif metric == "faithfulness":
            prompt = self._build_faithfulness_prompt(question, response, benchmark)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        try:
            if self.provider == "groq":
                # Use Groq API
                result = await asyncio.to_thread(
                    self._call_groq_api,
                    prompt
                )
                response_text = result
            else:
                # Use Gemini API
                result = await asyncio.to_thread(
                    self.model.generate_content,
                    prompt,
                    generation_config=self.generation_config
                )
                response_text = result.text
            
            # Parse response
            parsed = self._parse_json_response(response_text)
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Add metadata
            parsed["latency_ms"] = latency_ms
            parsed["model"] = self.model_name
            parsed["success"] = True
            
            # Estimate tokens (rough approximation)
            input_tokens = len(prompt.split()) * 1.3  # rough estimate
            output_tokens = len(response_text.split()) * 1.3
            parsed["input_tokens"] = int(input_tokens)
            parsed["output_tokens"] = int(output_tokens)
            
            return parsed
            
        except Exception as e:
            logger.error(f"Error evaluating with {metric}: {str(e)}")
            return {
                "score": 0.0,
                "label": "error",
                "reasoning": f"Evaluation failed: {str(e)}",
                "latency_ms": (time.time() - start_time) * 1000,
                "model": self.model_name,
                "success": False,
                "error": str(e)
            }
    
    def _call_groq_api(self, prompt: str) -> str:
        """Call Groq API"""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content
    
    async def evaluate_batch(
        self,
        metric: str,
        data: List[Tuple[str, str, str]],  # List of (question, response, benchmark)
        max_concurrent: int = 3
    ) -> List[Dict]:
        """
        Evaluate a batch of responses with rate limiting
        
        Args:
            metric: Metric to evaluate
            data: List of (question, response, benchmark) tuples
            max_concurrent: Maximum concurrent API calls
            
        Returns:
            List of evaluation results
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def evaluate_with_semaphore(item):
            async with semaphore:
                question, response, benchmark = item
                return await self.evaluate_single(metric, question, response, benchmark)
        
        tasks = [evaluate_with_semaphore(item) for item in data]
        results = await asyncio.gather(*tasks)
        
        return results


class EvaluationPipeline:
    """Main evaluation pipeline orchestrator"""
    
    def __init__(
        self,
        api_key: str,
        config: Dict,
        database=None
    ):
        """
        Initialize evaluation pipeline
        
        Args:
            api_key: API key (Gemini or Groq)
            config: Configuration dictionary
            database: Database instance for logging
        """
        provider = config.get("llm", {}).get("provider", "gemini")
        
        self.judge = LLMJudge(
            api_key=api_key,
            model=config.get("llm", {}).get("model", "gemini-2.0-flash-exp"),
            temperature=config.get("llm", {}).get("temperature", 0.2),
            max_tokens=config.get("llm", {}).get("max_tokens", 2048),
            provider=provider
        )
        self.config = config
        self.database = database
        self.batch_size = config.get("batch", {}).get("size", 5)
        self.max_concurrent = config.get("batch", {}).get("max_concurrent", 3)
    
    async def evaluate_dataset(
        self,
        metric: str,
        questions: List[str],
        responses: List[str],
        benchmarks: List[str],
        run_id: int = None,
        progress_callback=None
    ) -> List[Dict]:
        """
        Evaluate entire dataset for a specific metric
        
        Args:
            metric: Metric to evaluate
            questions: List of questions
            responses: List of responses
            benchmarks: List of benchmark answers
            run_id: Database run ID for logging
            progress_callback: Function to call with progress updates
            
        Returns:
            List of evaluation results
        """
        assert len(questions) == len(responses) == len(benchmarks)
        
        total = len(questions)
        results = []
        
        # Process in batches
        for i in range(0, total, self.batch_size):
            batch_end = min(i + self.batch_size, total)
            batch_data = list(zip(
                questions[i:batch_end],
                responses[i:batch_end],
                benchmarks[i:batch_end]
            ))
            
            # Evaluate batch
            batch_results = await self.judge.evaluate_batch(
                metric=metric,
                data=batch_data,
                max_concurrent=self.max_concurrent
            )
            
            # Log to database if available
            if self.database and run_id:
                for idx, result in enumerate(batch_results):
                    row_index = i + idx
                    self.database.add_evaluation_result(
                        run_id=run_id,
                        row_index=row_index,
                        question=questions[row_index],
                        response=responses[row_index],
                        benchmark_answer=benchmarks[row_index],
                        metric_name=metric,
                        score=result.get("score", 0.0),
                        label=result.get("label", "unknown"),
                        reasoning=result.get("reasoning", ""),
                        api_call_duration_ms=result.get("latency_ms", 0)
                    )
                    
                    # Log API usage
                    if result.get("success"):
                        self.database.log_api_usage(
                            run_id=run_id,
                            provider="gemini",
                            model=result.get("model", "unknown"),
                            input_tokens=result.get("input_tokens", 0),
                            output_tokens=result.get("output_tokens", 0),
                            total_tokens=result.get("input_tokens", 0) + result.get("output_tokens", 0),
                            latency_ms=result.get("latency_ms", 0),
                            success=1
                        )
            
            results.extend(batch_results)
            
            # Call progress callback
            if progress_callback:
                progress = batch_end / total
                progress_callback(progress, f"Processed {batch_end}/{total} rows")
        
        return results
