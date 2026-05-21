from typing import List, Dict, Any
import math

class ResultReranker:
    """Re-rank search results by relevance."""
    
    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot = sum(x*y for x,y in zip(a,b))
        norm_a = math.sqrt(sum(x*x for x in a))
        norm_b = math.sqrt(sum(x*x for x in b))
        return dot / (norm_a * norm_b) if norm_a * norm_b > 0 else 0.0
    
    @staticmethod
    def rerank_by_similarity(results: List[Dict[str, Any]], query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Re-rank results by similarity score."""
        for result in results:
            result["similarity_score"] = ResultReranker.cosine_similarity(
                query_embedding,
                result.get("embedding", [])
            )
        
        sorted_results = sorted(results, key=lambda x: x["similarity_score"], reverse=True)
        return sorted_results[:top_k]