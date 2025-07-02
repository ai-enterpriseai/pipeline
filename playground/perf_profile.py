import asyncio
import time
from pathlib import Path

try:
    from src.pipeline.processor import Processor
    from src.pipeline.embedder import Embedder
    from src.pipeline.indexer import Indexer
    from src.pipeline.utils.configs import (
        ProcessorConfig,
        EmbedderConfig,
        IndexerConfig,
    )
except ModuleNotFoundError as exc:
    missing = str(exc).split("No module named")[-1].strip().strip("'")
    print(
        f"Missing dependency: {missing}. Install requirements with 'pip install -r requirements.txt' "
        "and ensure all optional packages are available."
    )
    raise SystemExit(1) from exc

def create_dummy_files(path: Path, count: int = 300) -> None:
    """Create dummy text files for performance testing."""
    path.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        (path / f"file_{i}.txt").write_text(f"dummy content {i} " * 50)  # Make files larger for realistic testing

def measure_time(label: str):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"{label}: {end - start:.2f}s")

async def main():
    print("🚀 Starting Cogit Performance Assessment")
    print("=" * 50)
    
    tmp_dir = Path("temp_data")
    
    # Create dummy files
    print("📁 Creating test files...")
    create_dummy_files(tmp_dir)  # Run synchronously for simplicity
    print(f"✅ Created {len(list(tmp_dir.glob('*.txt')))} test files")

    # Configure components with mock values for testing
    processor_config = ProcessorConfig()
    embedder_config = EmbedderConfig()
    
    # Mock indexer config for testing (no actual DB connection needed)
    indexer_config = IndexerConfig(
        collection_name="test_collection",
        url="http://localhost:6333",  # Default Qdrant URL
        qdrant_api_key="mock_api_key",
        use_local_db=True,  # Use local for testing
        batch_size=50,  # Smaller batch for testing
    )

    processor = Processor(processor_config)
    embedder = Embedder(embedder_config)
    
    # Note: For this assessment, we'll focus on processor and embedder performance
    # Indexer requires actual database connection, so we'll test up to embedding generation
    
    total_start = time.perf_counter()
    
    try:
        # Stage 1: Document Loading
        print("\n📚 Stage 1: Document Loading")
        load_start = time.perf_counter()
        docs = await processor.load_documents(tmp_dir)
        load_end = time.perf_counter()
        load_time = load_end - load_start
        print(f"✅ Loaded {len(docs)} documents in {load_time:.2f}s")
        print(f"📊 Loading rate: {len(docs)/load_time:.1f} docs/sec")

        # Stage 2: Document Processing
        print("\n⚙️  Stage 2: Document Processing (chunking, deduplication)")
        process_start = time.perf_counter()
        processed = []
        async for doc in processor.process_documents(docs):
            if doc:
                processed.append(doc)
        process_end = time.perf_counter()
        process_time = process_end - process_start
        print(f"✅ Processed {len(processed)} documents in {process_time:.2f}s")
        print(f"📊 Processing rate: {len(processed)/process_time:.1f} docs/sec")

        # Stage 3: Embedding Generation (without database upload)
        print("\n🧠 Stage 3: Embedding Generation")
        embed_start = time.perf_counter()
        
        total_chunks = sum(len(doc.chunks) for doc in processed)
        embedded_count = 0
        
        for doc in processed:
            if doc.chunks:
                # Extract text from chunk tuples (text, metadata)
                texts = [chunk[0] for chunk in doc.chunks]  # chunk[0] is the text content
                dense_embeddings = await embedder.get_dense_embeddings(texts)
                
                # Generate sparse embeddings  
                sparse_embeddings = await embedder.get_sparse_embeddings(texts)
                
                embedded_count += len(texts)
        
        embed_end = time.perf_counter()
        embed_time = embed_end - embed_start
        print(f"✅ Generated embeddings for {embedded_count} chunks in {embed_time:.2f}s")
        print(f"📊 Embedding rate: {embedded_count/embed_time:.1f} chunks/sec")

        # Overall Performance Summary
        total_time = time.perf_counter() - total_start
        print("\n" + "=" * 50)
        print("📋 PERFORMANCE ASSESSMENT SUMMARY")
        print("=" * 50)
        print(f"🕐 Total pipeline time: {total_time:.2f}s")
        print(f"📄 Files processed: {len(docs)}")
        print(f"📝 Chunks generated: {total_chunks}")
        print(f"⚡ Overall throughput: {len(docs)/total_time:.1f} docs/sec")
        print("\n📊 Stage Breakdown:")
        print(f"  • Loading: {load_time:.2f}s ({load_time/total_time*100:.1f}%)")
        print(f"  • Processing: {process_time:.2f}s ({process_time/total_time*100:.1f}%)")
        print(f"  • Embedding: {embed_time:.2f}s ({embed_time/total_time*100:.1f}%)")
        
        # Performance Analysis
        print("\n🔍 PERFORMANCE ANALYSIS:")
        if load_time > total_time * 0.4:
            print("⚠️  Loading is a major bottleneck (>40% of total time)")
            print("   💡 Consider: async file loading, parallel processing")
        
        if embed_time > total_time * 0.5:
            print("⚠️  Embedding is a major bottleneck (>50% of total time)")
            print("   💡 Consider: batching, GPU acceleration, async processing")
            
        if total_time > 60:  # More than 1 minute for 300 files
            print("⚠️  Overall performance is slow")
            print("   💡 Consider: pipeline parallelization, caching, optimization")
        else:
            print("✅ Overall performance is acceptable")

    except Exception as e:
        print(f"❌ Error during assessment: {e}")
    finally:
        await processor.close()
        # Clean up test files
        import shutil
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
            print(f"🧹 Cleaned up test directory")

if __name__ == "__main__":
    asyncio.run(main())
