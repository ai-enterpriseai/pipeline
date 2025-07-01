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

async def create_dummy_files(path: Path, count: int = 300) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        (path / f"file_{i}.txt").write_text(f"dummy content {i}")

def measure_time(label: str):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"{label}: {end - start:.2f}s")

async def main():
    tmp_dir = Path("temp_data")
    await asyncio.to_thread(create_dummy_files, tmp_dir)

    processor = Processor(ProcessorConfig())
    embedder = Embedder(EmbedderConfig())
    indexer = Indexer(IndexerConfig(), embedder.dense_embedder, embedder.sparse_embedder)

    start = time.perf_counter()
    try:
        docs = await processor.load_documents(tmp_dir)
        load_end = time.perf_counter()
        print(f"Loaded {len(docs)} docs in {load_end - start:.2f}s")

        processed = []
        async for doc in processor.process_documents(docs):
            if doc:
                processed.append(doc)
        process_end = time.perf_counter()
        print(f"Processed {len(processed)} docs in {process_end - load_end:.2f}s")

        await indexer.index_documents(processed)
        index_end = time.perf_counter()
        print(f"Indexed {len(processed)} docs in {index_end - process_end:.2f}s")
    finally:
        await processor.close()
        await indexer.close()

if __name__ == "__main__":
    asyncio.run(main())
