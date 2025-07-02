#!/usr/bin/env python3
"""
Simple baseline test comparing DirectoryLoader performance.
Tests only the loading stage to isolate the improvement.
"""

import asyncio
import time
import tempfile
from pathlib import Path

try:
    from src.pipeline.processor import Processor
    from src.pipeline.utils.configs import ProcessorConfig
    from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
except ModuleNotFoundError as exc:
    missing = str(exc).split("No module named")[-1].strip().strip("'")
    print(
        f"Missing dependency: {missing}. Install requirements with 'pip install -r requirements.txt' "
        "and ensure all optional packages are available."
    )
    raise SystemExit(1) from exc

def create_test_files(directory: Path, count: int = 50) -> None:
    """Create test files for benchmarking."""
    directory.mkdir(exist_ok=True)
    
    for i in range(count):
        file_path = directory / f"test_file_{i:03d}.txt"
        content = f"This is test file {i}\n" + "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 20
        file_path.write_text(content)
    
    print(f"Created {count} test files in {directory}")

def test_original_directory_loader(directory: Path) -> float:
    """Test the original DirectoryLoader approach."""
    print("\n🔄 Testing Original DirectoryLoader...")
    
    start_time = time.time()
    
    loader = DirectoryLoader(
        str(directory),
        glob="**/*",
        loader_cls=UnstructuredFileLoader,
        use_multithreading=True, 
        silent_errors=True,
        show_progress=False,
    )
    documents = loader.load()
    
    end_time = time.time()
    duration = end_time - start_time
    rate = len(documents) / duration if duration > 0 else 0
    
    print(f"✅ Original DirectoryLoader:")
    print(f"   - Time: {duration:.2f}s")
    print(f"   - Documents: {len(documents)}")
    print(f"   - Rate: {rate:.1f} docs/sec")
    
    return duration

async def test_async_loading(directory: Path) -> float:
    """Test the new async loading approach."""
    print("\n🔄 Testing Async Loading...")
    
    config = ProcessorConfig(max_concurrent_files=5)
    processor = Processor(config)
    
    start_time = time.time()
    documents = await processor.load_documents(directory)
    end_time = time.time()
    
    duration = end_time - start_time
    rate = len(documents) / duration if duration > 0 else 0
    
    print(f"✅ Async Loading:")
    print(f"   - Time: {duration:.2f}s")
    print(f"   - Documents: {len(documents)}")
    print(f"   - Rate: {rate:.1f} docs/sec")
    
    await processor.close()
    return duration

async def run_comparison():
    """Run comparison between original and async loading."""
    print("🚀 Baseline Loading Performance Test")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_dir = Path(tmp_dir)
        
        # Create test files
        print("📁 Creating test files...")
        create_test_files(test_dir, count=50)
        
        # Test original approach
        original_time = test_original_directory_loader(test_dir)
        
        # Test async approach
        async_time = await test_async_loading(test_dir)
        
        # Compare results
        print("\n📊 Performance Comparison:")
        print("-" * 50)
        print(f"Original DirectoryLoader: {original_time:.2f}s")
        print(f"Async Loading:           {async_time:.2f}s")
        
        if original_time > 0:
            improvement = (original_time - async_time) / original_time * 100
            speedup = original_time / async_time if async_time > 0 else float('inf')
            print(f"Improvement:             {improvement:.1f}%")
            print(f"Speedup:                 {speedup:.1f}x")

if __name__ == "__main__":
    asyncio.run(run_comparison())