import json
import asyncio
import torch
import numpy as np

from typing import List, Optional, Dict, Any, Union
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from pinecone_text.sparse import BM25Encoder
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
)

from .utils.configs import EmbedderConfig, EmbedderType
from .utils.logging import setup_logger

logger = setup_logger(__name__)

class BaseEmbedder:
    """Base class for embedders."""
    
    def __init__(self, config: EmbedderConfig):
        self.config = config
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def embed_with_retry(self, texts: List[str]) -> Any:
        """
        Embed with retry mechanism.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Embeddings in appropriate format
            
        Raises:
            RetryError: If all retry attempts fail
        """
        return await self.embed(texts)
    
    async def embed(self, texts: List[str]) -> Any:
        """
        Abstract method for embedding generation.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Embeddings in appropriate format
        """
        raise NotImplementedError

class DenseEmbedder(BaseEmbedder):
    """Handles dense embeddings using sentence-transformers."""
    
    def __init__(self, config: EmbedderConfig):
        super().__init__(config)
        self.model = self._load_model()
        
    def _load_model(self) -> Union[SentenceTransformer, Any]:
        """
        Load the embedding model based on configuration.
        
        Returns:
            Loaded model
            
        Raises:
            RuntimeError: If model loading fails
        """
        try:
            if self.config.embedder_type == EmbedderType.SENTENCE_TRANSFORMER:
                logger.info(f"Loading sentence transformer model: {self.config.dense_model_name}")
                model = SentenceTransformer(
                    self.config.dense_model_name,
                    device=self.config.device
                )
                model.max_seq_length = self.config.max_length
                return model
            elif self.config.embedder_type == EmbedderType.OPENAI:
                logger.info("Loading OpenAI embeddings client")
                client = OpenAI()
                return client.embeddings
            else:
                raise ValueError(f"Unsupported embedder type: {self.config.embedder_type}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Failed to load model: {e}")

    async def embed1(self, texts: List[str]) -> torch.Tensor:
        """
        Generate dense embeddings.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Tensor of embeddings
            
        Raises:
            RuntimeError: If embedding generation fails
        """
        try:
            if self.config.embedder_type == EmbedderType.SENTENCE_TRANSFORMER:
                embeddings = await asyncio.to_thread(
                    self.model.encode,
                    texts,
                    batch_size=self.config.batch_size,
                    normalize_embeddings=self.config.normalize_embeddings,
                    show_progress_bar=True
                )
                logger.info("Sentence transformer embedding complete")
                return embeddings            
            elif self.config.embedder_type == EmbedderType.OPENAI:
                # Handle empty strings for OpenAI
                processed_texts = [t if t else " " for t in texts]
                response = await asyncio.to_thread(
                    self.model.create,
                    input=processed_texts,
                    model=self.config.dense_model_name
                )
                logger.info("OpenAI embedding complete")
                return [e.embedding for e in response.data]
        except Exception as e:
            logger.error(f"Dense embedding failed: {e}")
            raise RuntimeError(f"Dense embedding failed: {e}")

class SparseEmbedder(BaseEmbedder):
    """Handles sparse embeddings using BM25Encoder from pinecone-text."""
    
    def __init__(self, config: EmbedderConfig):
        super().__init__(config)
        if config.sparse_model_path.exists():
            # Load existing model if available
            self.model = BM25Encoder()
            self.model.load(str(config.sparse_model_path))
            self._is_fitted = True
        else:
            # Initialize with config parameters if provided, otherwise use defaults
            self.model = BM25Encoder(
                b=config.bm25_b,
                k1=config.bm25_k1
            )
            self._is_fitted = False

    # TODO not use right now 
    def _validate_vector(self, vector: Dict[str, List]) -> bool:
        """
        Validate sparse vector format.
        
        Args:
            vector: Sparse vector dictionary
            
        Returns:
            bool: True if vector is valid
        """
        try:
            indices = vector.get("indices", [])
            values = vector.get("values", [])
            
            return (
                bool(indices) and 
                bool(values) and 
                len(indices) == len(values) and 
                all(isinstance(i, int) for i in indices) and 
                all(isinstance(v, (int, float)) for v in values)
            )
        except Exception as e:
            logger.error(f"Vector validation failed: {e}")
            return False
        
    async def fit(self, texts: List[str]) -> None:
        """
        Fit BM25 model on corpus.
        
        Args:
            texts: List of texts to fit on
            
        Raises:
            RuntimeError: If fitting fails
        """
        try:
            # Preprocess texts to handle empty/whitespace cases
            processed_texts = [text.strip() if text else "empty_document" for text in texts]

            await asyncio.to_thread(self.model.fit, processed_texts)
            await self._save_params()
            self._is_fitted = True
            logger.info("Sparse model has been fit")
        except Exception as e:
            logger.error(f"Failed to fit BM25: {e}")
            raise RuntimeError(f"Failed to fit BM25: {e}")

    async def embed(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Generate sparse embeddings.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of sparse embedding dictionaries
            
        Raises:
            ValueError: If model not fitted
            RuntimeError: If embedding generation fails
        """
        if not self._is_fitted:
            await self.fit(texts)
            
        try:
            # Preprocess texts to handle empty/whitespace cases
            processed_texts = [text.strip() if text else "empty_document" for text in texts]

            vectors = await asyncio.to_thread(
                self.model.encode_documents,
                processed_texts
            )

            logger.info("Sparse embedding complete.")                
            return [{
                "indices": vector["indices"] if isinstance(vector, dict) else vector.indices.tolist(),
                "values": vector["values"] if isinstance(vector, dict) else vector.values.tolist()
            } for vector in vectors]
            
        except Exception as e:
            logger.error(f"Sparse embedding failed: {e}")
            raise RuntimeError(f"Sparse embedding failed: {e}")

    async def _save_params(self) -> None:
        """
        Save BM25 parameters to file.
        
        Raises:
            RuntimeError: If saving fails
        """
        try:
            await asyncio.to_thread(self.model.dump, str(self.config.sparse_model_path))
        except Exception as e:
            logger.error(f"Failed to save BM25 parameters: {e}")
            raise RuntimeError(f"Failed to save BM25 parameters: {e}")

    async def _load_params(self) -> None:
        """
        Load BM25 parameters from file.
        
        Raises:
            RuntimeError: If loading fails
        """
        try:
            async with asyncio.Lock():
                with open(self.config.sparse_model_path, 'r') as f:
                    params = json.load(f)
                    
            self.model = BM25Encoder()
            self.model.idf_ = np.array(params["idf"])
            self.model.vocabulary_ = params["vocabulary"]
            
        except Exception as e:
            logger.error(f"Failed to load BM25 parameters: {e}")
            raise RuntimeError(f"Failed to load BM25 parameters: {e}")
        

class Embedder:
    """Main embedding class for generating dense and sparse embeddings."""
    
    def __init__(self, config: Optional[EmbedderConfig] = None):
        self.config = config or EmbedderConfig()
        self.dense_embedder = DenseEmbedder(self.config)
        self.sparse_embedder = SparseEmbedder(self.config)
        
    async def get_dense_embeddings(self, texts: List[str]) -> torch.Tensor:
        """
        Generate dense embeddings for texts.
        
        Args:pytest
            texts: List of texts to embed
            
        Returns:
            Tensor of dense embeddings
        """
        try:
            return await self.dense_embedder.embed_with_retry(texts)
        except Exception as e:
            logger.error(f"Dense embedding failed: {e}")
            raise RuntimeError(f"Dense embedding failed: {e}")

    async def get_sparse_embeddings(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Generate sparse embeddings for texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of sparse embedding dictionaries
        """
        try:
            await self.sparse_embedder.fit(texts)  # Fit BM25 on the corpus
            return await self.sparse_embedder.embed_with_retry(texts)
        except Exception as e:
            logger.error(f"Sparse embedding failed: {e}")
            raise RuntimeError(f"Sparse embedding failed: {e}")        