# pipeline
pipeline is a python package that provides a pipeline for efficient document retrieval and reranking. it leverages vector embeddings, approximate nearest neighbor search, and reranking models to retrieve relevant documents for a given query. the package also includes utilities for query decomposition, caching, and asynchronous processing.

# features
* **vector indexing**: index documents using dense and sparse vector embeddings for efficient similarity search.
* **retrieval**: retrieve relevant documents for a given query using a combination of dense and sparse vector search.
* **query decomposition**: optionally decompose complex queries into sub-queries for improved retrieval.
* **reranking**: rerank retrieved documents using a reranking model to improve relevance.
* **caching**: cache query results for faster retrieval of frequently asked queries.
* **asynchronous processing**: utilize asynchronous processing for improved performance.
* **logging**: comprehensive logging for debugging and monitoring.