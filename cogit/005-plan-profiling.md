# Plan 005: Profiling and Benchmarking

## Objective
Establish repeatable profiling scripts to quantify performance improvements over time.

## Current Status
Timing metrics are collected manually. There is no automated benchmark to compare changes.

## Proposed Approach
1. Create a small sample repository to act as benchmarking data.
2. Write a script that runs the full indexing pipeline and records stage timings.
3. Integrate with CI to run benchmarks on demand.

## Dependencies
- Plan 001: Performance Analysis

## Verification
- Running the script outputs timing data for each pipeline stage.
- Results can be compared between commits.

## Acceptance Criteria
- Profiling script exists and generates reproducible metrics.
- Metrics stored in a log or file for historical comparison.

## Estimated Effort
1–2 days for scripting and integration.
