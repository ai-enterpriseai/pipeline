# Plan 002: Async File Loading

## Objective
Implement asynchronous file loading to reduce I/O wait times during indexing.

## Current Status
`DirectoryLoader` loads files sequentially and uses `asyncio.to_thread` for blocking I/O, limiting concurrency.

## Proposed Approach
1. Replace `DirectoryLoader` with an implementation using `aiofiles` or similar async library.
2. Spawn tasks for parsing files concurrently using `asyncio.TaskGroup`.
3. Maintain existing rate limiting but adjust thresholds for disk throughput.

## Dependencies
- Plan 001: Performance Analysis

## Verification
- Measure time taken to load a fixed number of files before and after the change.
- Ensure loaded documents match original content.

## Acceptance Criteria
- File loading step executes concurrently and shows measurable speedup.
- No regressions in document content or order.

## Estimated Effort
2–3 days for implementation and testing.
