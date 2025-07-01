# Plan 004: Database Upload Optimization

## Objective
Reduce indexing time by combining uploads from multiple documents and tuning upload concurrency.

## Current Status
`Indexer.index_documents` uploads each document's vectors separately. Small batches result in frequent database calls.

## Proposed Approach
1. Buffer vectors from several documents before uploading to the database.
2. Expose configuration for batch size and parallel uploads.
3. Profile memory usage to prevent buffer overflows for large datasets.

## Dependencies
- Plan 001: Performance Analysis
- Plan 003: Embedding Batching

## Verification
- Measure the number of upload calls and overall indexing time before and after changes.
- Validate that all expected vectors appear in the database.

## Acceptance Criteria
- Upload step sends fewer, larger requests without data loss.
- Indexing time decreases or remains stable with higher throughput.

## Estimated Effort
2 days for development and benchmarking.
