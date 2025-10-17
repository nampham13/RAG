"""
Query Expansion Module
======================
Mở rộng queries để cải thiện embedding similarity.
"""

class QueryExpander:
    """Mở rộng queries với context thêm để cải thiện retrieval."""
    
    def __init__(self):
        # Dictionary mapping query keywords to expanded descriptions
        self.expansions = {
            "query rewriting": "Query Rewriting transforms ambiguous poorly structured queries into clarified intent breaks complex questions sub-queries adds precision",
            "page index": "PageIndex builds hierarchical tree document human-like navigation reasoning sections",
            "graph rag": "Graph RAG knowledge graph documents connecting entities concepts facts multi-hop reasoning",
            "self-reasoning": "Self-Reasoning LLM evaluates retrieved chunks active quality inspector",
            "query routing": "Query Routing complexity-based routing strategy retrieval efficiency",
            "bm25": "BM25 exact matching keyword search integration combination",
            "rag pipeline": "RAG retrieval-augmented generation techniques accuracy",
        }
    
    def expand(self, query: str) -> str:
        """
        Mở rộng query nếu match được keywords.
        
        Args:
            query: Original query
            
        Returns:
            Expanded query or original if no match
        """
        query_lower = query.lower()
        
        # Kiểm tra exact match trước
        if query_lower in self.expansions:
            return f"{query} {self.expansions[query_lower]}"
        
        # Kiểm tra partial match
        for keyword, expansion in self.expansions.items():
            if keyword in query_lower:
                return f"{query} {expansion}"
        
        # Nếu không match, trả lại query gốc
        return query


# Test
if __name__ == "__main__":
    expander = QueryExpander()
    
    queries = [
        "Query Rewriting",
        "Query Rewriting meaning",
        "What is PageIndex",
        "tell me about RAG pipeline"
    ]
    
    for q in queries:
        expanded = expander.expand(q)
        print(f"Original: {q}")
        print(f"Expanded: {expanded}")
        print()