import asyncio
import torch 

from typing import List, Dict, Any, Optional, Tuple, Generic, TypeVar, Union
from datetime import datetime, timedelta

from collections import OrderedDict

from pydantic import BaseModel, Field, ConfigDict
from qdrant_client import models
from rerankers import Reranker
import cohere

from .indexer import Indexer
from .utils.configs import RetrieverConfig, RerankerType
from .utils.model import QueryDecomposition
from .utils.logging import setup_logger
from .utils.types import CacheEntry, SearchResult, ChatMessage, DecomposedQuery

logger = setup_logger(__name__)

T = TypeVar('T')

class LRUCache(BaseModel, Generic[T]):
    """LRU Cache with expiration."""
    max_size: int = Field(default=1000)
    ttl_seconds: int = Field(default=3600)
    cache: Dict[str, CacheEntry[T]] = Field(default_factory=OrderedDict)
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def get(self, key: str) -> Optional[T]:
        """Get value if exists and not expired."""
        if key not in self.cache:
            return None
            
        entry = self.cache[key]
        if entry.is_expired:
            del self.cache[key]
            return None
            
        # Update LRU order
        value = self.cache.pop(key)
        self.cache[key] = value
        return entry.value
        
    def put(self, key: str, value: T) -> None:
        """Add value to cache with expiration."""
        entry = CacheEntry(
            value=value,
            expiry=datetime.now() + timedelta(seconds=self.ttl_seconds)
        )
        
        if key in self.cache:
            del self.cache[key]
        elif len(self.cache) >= self.max_size:
            # Remove oldest item
            self.cache.pop(next(iter(self.cache)))
            
        self.cache[key] = entry

class Retriever:
    """Handles document retrieval and reranking."""
    
    def __init__(
        self,
        config: RetrieverConfig,
        vector_index: Indexer,
        llm_client = QueryDecomposition # TODO can be set as QueryDecomposition at runtime 
    ):
        """Initialize retriever with models and config."""
        self.config = config
        self.vector_index = vector_index
        self.dense_embedder = self.vector_index.dense_embedder
        self.sparse_embedder = self.vector_index.sparse_embedder
        self.reranker = self._initialize_reranker()
        self.llm_client = llm_client
        self.cache = LRUCache[List[SearchResult]](
            max_size=config.cache.max_size,
            ttl_seconds=config.cache.ttl
        )
        
    def _initialize_reranker(self) -> Union[Reranker, cohere.AsyncClientV2]:
        """Initialize reranker model based on configuration."""
        try:
            if self.config.retriever.reranker.reranker_type == RerankerType.RERANKER:
                logger.info(f"Loading cross-encoder model: {self.config.retriever.reranker.model_name}")
                return Reranker(self.config.retriever.reranker.model_name)
            
            elif self.config.retriever.reranker.reranker_type == RerankerType.COHERE:
                if not self.config.retriever.reranker.api_key:
                    raise ValueError("API key required for Cohere reranker")
                logger.info("Initializing Cohere client")
                return cohere.AsyncClientV2(self.config.retriever.reranker.api_key)
                
        except Exception as e:
            logger.error(f"Failed to initialize reranker: {e}")
            raise RuntimeError(f"Reranker initialization failed: {e}")

    async def _preprocess_query(
        self, 
        query: str
    ) -> str:
        """Preprocess and validate query."""
        if not query.strip():
            raise ValueError("Empty query")
            
        query = query[:self.config.query_max_length].strip()
        return query

    async def _embed_query(
        self,
        query: str
    ) -> Tuple[torch.Tensor, Optional[Dict[str, List[int]]]]:
        """Generate query embeddings."""
        try:
            # Generate dense embedding
            dense_vector = await self.dense_embedder.embed([query])
            
            # Generate sparse embedding
            try:
                sparse_vector = await self.sparse_embedder.embed([query])
                sparse_vector = sparse_vector[0]
            except Exception as e:
                logger.warning(f"Sparse embedding failed: {e}")
                sparse_vector = None
                
            return dense_vector[0], sparse_vector
            
        except Exception as e:
            logger.error(f"Query embedding failed: {e}")
            raise RuntimeError(f"Query embedding failed: {e}")

    async def _search(
        self,
        dense_vector: torch.Tensor,
        sparse_vector: Optional[Dict[str, List[int]]] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[models.ScoredPoint]:
        """Execute search with vectors."""
        try:
            # Prepare prefetch
            prefetch = [
                models.Prefetch(
                    query=dense_vector,#.tolist(),
                    using="dense",
                    limit=self.config.top_k
                )
            ]
            
            if sparse_vector is not None:
                prefetch.append(
                    models.Prefetch(
                        query=models.SparseVector(**sparse_vector),
                        using="sparse",
                        limit=self.config.top_k
                    )
                )
            
            # Execute search
            results = await asyncio.to_thread(
                self.vector_index.client.query_points,
                collection_name=self.vector_index.config.collection_name,
                prefetch=prefetch,
                query=models.FusionQuery(
                    fusion=models.Fusion.RRF
                ),
                limit=self.config.top_k,
                with_payload=True,
                query_filter=filters
            )
            logger.info("Search complete.")
            return results.points
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise RuntimeError(f"Search failed: {e}")

    async def _rerank_results(
        self,
        query: str,
        results: List[models.ScoredPoint]
    ) -> List[SearchResult]:
        """Rerank search results based on configured reranker type."""
        try:
            # Prepare documents and metadata for reranking
            docs = []
            metadata = []
            for r in results:
                docs.append(r.payload.get('chunk_text', ''))
                metadata.append(r.payload.get('metadata', {}))

            # Perform reranking based on reranker type
            if self.config.retriever.reranker.reranker_type == RerankerType.COHERE:
                reranked = await asyncio.to_thread(
                    self.reranker.rerank,
                    query=query,
                    documents=docs,
                    top_n=self.config.retriever.reranker.top_k,
                    model=self.config.retriever.reranker.model_name
                )
                # Process Cohere results
                top_results = sorted(
                    reranked.results,
                    key=lambda x: x.relevance_score,
                    reverse=True
                )[:self.config.retriever.reranker.top_k]
                
                return [
                    SearchResult(
                        text=results[r.index].payload.get('chunk_text', ''),
                        score=r.relevance_score,
                        metadata=results[r.index].payload.get('metadata', {}),
                        original_rank=r.index,
                        reranked_score=r.relevance_score
                    )
                    for r in top_results
                    if r.index < len(results)
                ]
            else:
                # Handle other reranker types (rerankers lib)
                reranked = await asyncio.to_thread(
                    self.reranker.rank_async,
                    query=query,
                    docs=docs,
                    metadata=metadata
                )
                
                return [
                    SearchResult(
                        text=results[r.doc_id].payload.get('chunk_text', ''),
                        score=r.score,
                        metadata=results[r.doc_id].payload.get('metadata', {}),
                        original_rank=r.doc_id,
                        reranked_score=r.score
                    )
                    for r in reranked.results[:self.config.retriever.reranker.top_k]
                ]
                
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            raise RuntimeError(f"Reranking failed: {e}")
        
    async def retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> List[SearchResult]:
        """
        Retrieve and rerank documents for query.
        
        Args:
            query: Search query
            filters: Optional metadata filters
            use_cache: Whether to use query cache
            
        Returns:
            List of search results
            
        Raises:
            ValueError: If query is invalid
            RuntimeError: If retrieval fails
        """
        if not query.strip():
            raise ValueError("Empty query")

        try:
            # Check cache
            cache_key = f"{query}:{str(filters)}"
            if use_cache:
                cached = self.cache.get(cache_key)
                if cached:
                    logger.info("Using cached results")
                    return cached
            
            # Preprocess query
            processed_query = await self._preprocess_query(query)

            # Expand query if enabled
            # TODO decompose_query doesn't work due to issues with response_models in the instructor library, needs fixing 
            if self.config.decompose_query:
                decomposed = await self.llm_client.generate(ChatMessage.processed_query, DecomposedQuery)
                logger.info(f"Expanded query into {len(decomposed.sub_queries)} sub-queries")
                sub_queries = decomposed.sub_queries
                reasoning = decomposed.reasoning # TODO maybe embed reasoning too? It can also be changed to reflect a higher level concept, and serve as meta-querying 
            else:
                decomposed = DecomposedQuery(
                    original_query=processed_query,
                    sub_queries=[""],
                    reasoning="Query expansion disabled"
                )
                sub_queries = decomposed.sub_queries
                reasoning = decomposed.reasoning 

            all_results = []
            for sub_query in sub_queries:
                # Generate embeddings
                dense_vector, sparse_vector = await self._embed_query(sub_query)
                
                # Execute search
                results = await self._search(
                    dense_vector,
                    sparse_vector,
                    filters
                )
                all_results.extend(results)
            
            # Remove duplicates and take top_k
            unique_results = self._deduplicate_results(
                all_results,
                self.config.top_k
            )
            
            # Rerank results
            final_results = await self._rerank_results(
                processed_query,
                unique_results
            )
            
            # Update cache
            if use_cache:
                self.cache.put(cache_key, final_results)
            
            return final_results
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            raise RuntimeError(f"Retrieval failed: {e}")

    async def batch_retrieve(
        self,
        queries: List[str],
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[SearchResult]]:
        """
        Retrieve documents for multiple queries.
        
        Args:
            queries: List of search queries
            filters: Optional metadata filters
            
        Returns:
            Dictionary mapping queries to results
        """
        results = {}
        for query in queries:
            try:
                results[query] = await self.retrieve(
                    query,
                    filters,
                    use_cache=True
                )
            except Exception as e:
                logger.error(f"Batch retrieval failed for query '{query}': {e}")
                results[query] = []
        return results

    def _deduplicate_results(
        self,
        results: List[models.ScoredPoint],
        limit: int
    ) -> List[models.ScoredPoint]:
        """Remove duplicate results keeping highest scores."""
        seen = set()
        unique = []
        
        for result in sorted(results, key=lambda x: x.score, reverse=True):
            text = result.payload.get('chunk_text', '')
            if text not in seen:
                seen.add(text)
                unique.append(result)
                if len(unique) >= limit:
                    break
                    
        return unique