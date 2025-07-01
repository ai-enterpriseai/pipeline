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

### Initial Profiling Attempt
An initial run of `scripts/perf_profile.py` failed because required
dependencies such as `langchain_community` and `unstructured` are not
installed in the current environment. Profiling cannot proceed until these
packages are available.
