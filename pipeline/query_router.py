"""
Adaptive Query Router
====================
Routes queries based on complexity to determine retrieval strategy:
- Simple factual queries → single-step retrieval
- Complex analytical queries → multi-step iterative retrieval  
- General conversation → no retrieval
"""

import logging
from typing import Literal, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

QueryType = Literal["simple_factual", "complex_analytical", "general_conversation"]


@dataclass
class QueryAnalysis:
    """Analysis result for a query"""
    query_type: QueryType
    reasoning: str
    confidence: float
    requires_retrieval: bool
    suggested_top_k: int


class QueryRouter:
    """
    Analyzes query complexity and routes to appropriate retrieval strategy.
    Uses LLM-based classification for intelligent routing.
    """
    
    def __init__(self, llm_callable=None):
        """
        Initialize QueryRouter
        
        Args:
            llm_callable: Optional callable that takes messages and returns response
                         Format: llm_callable(messages: List[Dict]) -> Dict[str, Any]
        """
        self.llm_callable = llm_callable
        
    def analyze_query(self, query: str) -> QueryAnalysis:
        """
        Analyze query to determine routing strategy
        
        Args:
            query: User query text
            
        Returns:
            QueryAnalysis with routing decision
        """
        if self.llm_callable:
            return self._llm_based_analysis(query)
        else:
            return self._rule_based_analysis(query)
    
    def _llm_based_analysis(self, query: str) -> QueryAnalysis:
        """Use LLM to classify query complexity"""
        
        system_prompt = """You are a query analysis assistant. Classify the given query into one of three categories:

1. **simple_factual**: Direct factual questions that can be answered with specific information
   - Examples: "What is X?", "When did Y happen?", "Who is Z?"
   
2. **complex_analytical**: Questions requiring analysis, comparison, or synthesis of multiple pieces of information
   - Examples: "Compare X and Y", "Analyze the impact of...", "What are the implications..."
   
3. **general_conversation**: Casual conversation, greetings, or queries that don't require document retrieval
   - Examples: "Hello", "How are you?", "Thank you"

Respond ONLY with a JSON object in this exact format:
{
    "query_type": "simple_factual|complex_analytical|general_conversation",
    "reasoning": "Brief explanation",
    "confidence": 0.0-1.0,
    "requires_retrieval": true|false,
    "suggested_top_k": 5-20
}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Classify this query: {query}"}
        ]
        
        try:
            result = self.llm_callable(messages)
            response_text = result.get("response", "")
            
            # Parse JSON response
            import json
            # Extract JSON from response (handle potential markdown formatting)
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                analysis_dict = json.loads(json_str)
                
                return QueryAnalysis(
                    query_type=analysis_dict["query_type"],
                    reasoning=analysis_dict["reasoning"],
                    confidence=analysis_dict["confidence"],
                    requires_retrieval=analysis_dict["requires_retrieval"],
                    suggested_top_k=analysis_dict["suggested_top_k"]
                )
            else:
                logger.warning("No JSON found in LLM response, falling back to rule-based")
                return self._rule_based_analysis(query)
                
        except Exception as e:
            logger.warning(f"LLM-based analysis failed: {e}, falling back to rule-based")
            return self._rule_based_analysis(query)
    
    def _rule_based_analysis(self, query: str) -> QueryAnalysis:
        """Fallback rule-based classification"""
        
        query_lower = query.lower().strip()
        
        # General conversation patterns
        greeting_patterns = ['hello', 'hi', 'hey', 'thank', 'bye', 'good morning', 
                           'good afternoon', 'how are you']
        if any(pattern in query_lower for pattern in greeting_patterns):
            return QueryAnalysis(
                query_type="general_conversation",
                reasoning="Detected greeting or casual conversation",
                confidence=0.9,
                requires_retrieval=False,
                suggested_top_k=0
            )
        
        # Complex analytical patterns
        complex_patterns = ['compare', 'analyze', 'evaluate', 'assess', 'explain how',
                          'what are the implications', 'discuss', 'elaborate',
                          'what factors', 'why does', 'how does', 'relationship between']
        if any(pattern in query_lower for pattern in complex_patterns):
            return QueryAnalysis(
                query_type="complex_analytical",
                reasoning="Detected analytical/comparison keywords",
                confidence=0.85,
                requires_retrieval=True,
                suggested_top_k=15
            )
        
        # Simple factual patterns (default for queries with question marks or "what/who/when/where")
        factual_patterns = ['what is', 'who is', 'when', 'where', 'define', 'meaning of']
        if any(pattern in query_lower for pattern in factual_patterns) or '?' in query:
            return QueryAnalysis(
                query_type="simple_factual",
                reasoning="Detected factual question pattern",
                confidence=0.8,
                requires_retrieval=True,
                suggested_top_k=5
            )
        
        # Default: treat as simple factual if it looks like a question
        return QueryAnalysis(
            query_type="simple_factual",
            reasoning="Default classification for queries",
            confidence=0.6,
            requires_retrieval=True,
            suggested_top_k=5
        )