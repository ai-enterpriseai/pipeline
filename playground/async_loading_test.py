#!/usr/bin/env python3
"""
Test script for async loading performance comparison.
Compares old DirectoryLoader vs new async implementation.
"""

import asyncio
import time
import tempfile
from pathlib import Path

try:
    from src.pipeline.processor import Processor
    from src.pipeline.utils.configs import ProcessorConfig
except ModuleNotFoundError as exc:
    missing = str(exc).split("No module named")[-1].strip().strip("'")
    print(
        f"Missing dependency: {missing}. Install requirements with 'pip install -r requirements.txt' "
        "and ensure all optional packages are available."
    )
    raise SystemExit(1) from exc

def create_test_files(directory: Path, count: int = 100) -> None:
    """Create test files for benchmarking."""
    directory.mkdir(exist_ok=True)
    
    for i in range(count):
        file_path = directory / f"test_file_{i:03d}.txt"
        content = f"This is test file {i}\n" + "Lorem ipsum " * 50
        file_path.write_text(content)
    
    print(f"Created {count} test files in {directory}")

async def benchmark_loading(processor: Processor, directory: Path, label: str) -> float:
    """Benchmark document loading and return time taken."""
    print(f"\n🔄 Testing {label}...")
    
    start_time = time.time()
    documents = await processor.load_documents(directory)
    end_time = time.time()
    
    duration = end_time - start_time
    rate = len(documents) / duration if duration > 0 else 0
    
    print(f"✅ {label}:")
    print(f"   - Time: {duration:.2f}s")
    print(f"   - Documents: {len(documents)}")
    print(f"   - Rate: {rate:.1f} docs/sec")
    
    return duration

async def run_comparison():
    """Run performance comparison between old and new implementations."""
    print("🚀 Async Loading Performance Test")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_dir = Path(tmp_dir)
        
        # Create test files
        print("📁 Creating test files...")
        create_test_files(test_dir, count=100)
        
        # Test configurations
        configs = [
            (ProcessorConfig(max_concurrent_files=1), "Sequential (1 concurrent)"),
            (ProcessorConfig(max_concurrent_files=5), "Async (5 concurrent)"),
            (ProcessorConfig(max_concurrent_files=10), "Async (10 concurrent)"),
            (ProcessorConfig(max_concurrent_files=20), "Async (20 concurrent)"),
        ]
        
        results = {}
        
        for config, label in configs:
            processor = Processor(config)
            try:
                duration = await benchmark_loading(processor, test_dir, label)
                results[label] = duration
            finally:
                await processor.close()
        
        # Display comparison
        print("\n📊 Performance Comparison:")
        print("-" * 50)
        
        baseline = results.get("Sequential (1 concurrent)")
        if baseline:
            for label, duration in results.items():
                if label != "Sequential (1 concurrent)":
                    improvement = (baseline - duration) / baseline * 100
                    print(f"{label:25} | {duration:6.2f}s | {improvement:+5.1f}% vs sequential")
                else:
                    print(f"{label:25} | {duration:6.2f}s | baseline")
        
        # Find best performer
        best_config = min(results.keys(), key=lambda k: results[k])
        best_time = results[best_config]
        print(f"\n🏆 Best performer: {best_config} ({best_time:.2f}s)")
        
        if baseline and best_time:
            total_improvement = (baseline - best_time) / baseline * 100
            print(f"📈 Overall improvement: {total_improvement:.1f}%")

if __name__ == "__main__":
    asyncio.run(run_comparison())