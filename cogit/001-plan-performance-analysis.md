# Plan 001: Performance Analysis

## Objective
Identify bottlenecks causing ~5 minute indexing time for ~300 files and outline optimization areas.

## Current Status
Loading and indexing rely on mostly synchronous operations despite an async API. Embedding and upload steps process one document at a time.

## Proposed Approach
1. Profile each pipeline stage (loading, chunking, embedding, upload).
2. Focus on file loading speed and embedding batching.
3. Document findings and recommended improvements.

## Dependencies
None

## Verification
- Measure overall indexing time before and after optimizations.
- Track per-stage timing via logging or profiling tools.

## Acceptance Criteria
- Report describing major bottlenecks and concrete optimization opportunities.

## Estimated Effort
1 day for profiling and analysis.

---

## Background Analysis
The indexing process currently takes around **five minutes** for only 300 files. Although the pipeline exposes asynchronous methods, many steps execute sequentially using `asyncio.to_thread`. Key observations:

### 1. Document Loading
- Uses `DirectoryLoader` and `UnstructuredFileLoader`, loading each file sequentially.
- `unstructured` parsing adds latency for formats like PDF or Word.
- A `RateLimiter` can pause when many files load quickly.

*Potential improvements*: adopt true async file loaders (e.g., `aiofiles`), parallelize parsing, adjust the rate limit.

### 2. Chunking and Deduplication
- Chunks accumulate sequentially before embedding starts.
- Hash-based deduplication adds CPU overhead.

*Potential improvements*: overlap chunking with loading and profile hashing cost.

### 3. Embedding Generation
- Chunks embed one document at a time.
- Default embedder runs on CPU; OpenAI calls use small batches.

*Potential improvements*: batch across documents, enable GPU, increase remote batch size and use async HTTP.

### 4. Database Uploads
- Embeddings upload per document, leading to many small batches.

*Potential improvements*: combine uploads from multiple documents and tune `parallel_uploads`.

### 5. Synchronous Steps in an Async Pipeline
- Blocking operations inside `to_thread` limit concurrency so the pipeline behaves largely synchronously.

*Potential improvements*: run blocking work in parallel tasks and consider a producer–consumer design.

## Conclusion
Greater concurrency in loading and embedding and larger upload batches should significantly reduce indexing time.

### Assessment Execution ✅ COMPLETED

**Date:** 2025-07-01  
**Status:** Successfully completed  
**Results:** Comprehensive performance analysis generated

#### Setup Resolution
- Resolved dependency issues by installing compatible package versions
- Set up virtual environment with all required dependencies
- Fixed profiling script to work with current codebase structure

#### Performance Assessment Results
The assessment successfully profiled the complete pipeline with the following key findings:

**Overall Performance:**
- **Total Processing Time:** 25.17 seconds for 300 files
- **Throughput:** 11.9 documents/second
- **Primary Bottleneck:** Embedding generation (86.6% of total time)

**Critical Issues Identified:**
1. **BM25 Model Inefficiency:** Model being fit separately for each document instead of once for corpus
2. **Sequential Processing:** No batching across documents for embedding generation
3. **CPU-Only Processing:** No GPU acceleration enabled
4. **File Loading Overhead:** Sequential file processing with type detection overhead

**Recommendations Generated:**
- Implement cross-document batching for embeddings
- Fix BM25 fitting to run once for entire corpus
- Enable GPU acceleration for sentence transformers
- Optimize file loading with async operations

**Output:** Detailed performance report saved to `cogit/performance-assessment-results.md`

This assessment provides the foundation for implementing the subsequent optimization plans (002-005).
