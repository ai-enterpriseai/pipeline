# Plan 003: Embedding Batching

## Objective
Improve embedding throughput by batching chunks across documents and enabling parallel execution.

## Current Status
Embeddings are generated one document at a time. For remote models, batch size is small and HTTP requests are sequential.

## Proposed Approach
1. Aggregate chunks from multiple documents into larger batches.
2. Use asynchronous HTTP client for remote embeddings to overlap network latency.
3. Allow configuring batch size via settings.
4. If using SentenceTransformers locally, enable GPU execution when available.

## Dependencies
- Plan 001: Performance Analysis

## Verification
- Compare total embedding time for a fixed dataset before and after batching changes.
- Ensure embeddings remain deterministic for the same input.

## Acceptance Criteria
- Embedding stage processes batches rather than individual documents.
- Pipeline demonstrates reduced embedding time with correct outputs.

## Estimated Effort
2–4 days including implementation and profiling.
