#!/usr/bin/env python3
"""
Build FAISS index for Einstein semantic search.
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def build_faiss_index():
    """Build or rebuild the FAISS index for Einstein."""

    print("🏗️ Building Einstein FAISS Index")
    print("=" * 40)

    try:
        # Import Einstein
        import faiss

        from einstein.unified_index import EinsteinIndexHub

        # Initialize Einstein
        project_root = Path.cwd()
        print(f"📁 Project root: {project_root}")

        print("🔧 Initializing Einstein...")
        einstein = EinsteinIndexHub(project_root)

        # Check if embedding pipeline is working
        print("🧪 Testing embedding pipeline...")
        pipeline_ready = await einstein._ensure_embedding_pipeline()
        if not pipeline_ready:
            print("❌ Embedding pipeline not available - cannot build FAISS index")
            return False

        print("✅ Embedding pipeline ready")

        # Collect Python files to index
        print("📂 Collecting Python files...")
        python_files = list(project_root.rglob("*.py"))
        print(f"   Found {len(python_files)} Python files")

        # Sample files for testing (start with smaller set)
        max_files = min(100, len(python_files))  # Start with 100 files for testing
        sample_files = python_files[:max_files]
        print(f"   Processing {len(sample_files)} files for initial index")

        # Create embeddings
        print("🧠 Creating embeddings...")
        embeddings = []
        file_metadata = []

        start_time = time.time()

        for i, file_path in enumerate(sample_files):
            try:
                if i % 10 == 0:
                    print(f"   Processing {i+1}/{len(sample_files)}: {file_path.name}")

                # Read file content
                with open(file_path, encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                # Skip empty files
                if not content.strip():
                    continue

                # Take first chunk of content for embedding (to avoid token limits)
                content_chunk = content[:2000]  # Limit to ~2000 chars

                # Get embedding using Einstein's pipeline
                embedding, token_count = await einstein._safe_get_query_embedding(
                    content_chunk
                )

                if embedding is not None and embedding.size > 0:
                    embeddings.append(embedding)
                    file_metadata.append(
                        {
                            "file_path": str(file_path.relative_to(project_root)),
                            "absolute_path": str(file_path),
                            "token_count": token_count,
                            "content_preview": content_chunk[:200],
                        }
                    )

                # Limit processing time for testing
                if time.time() - start_time > 60:  # 1 minute limit for testing
                    print(f"   Time limit reached, processed {len(embeddings)} files")
                    break

            except Exception as e:
                logger.warning(f"Failed to embed {file_path}: {e}")
                continue

        if not embeddings:
            print("❌ No embeddings created - cannot build FAISS index")
            return False

        print(f"✅ Created {len(embeddings)} embeddings")

        # Convert to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)
        print(f"📊 Embeddings shape: {embeddings_array.shape}")

        # Create FAISS index
        print("🔍 Building FAISS index...")
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatL2(dimension)  # Simple L2 distance index

        # Add embeddings to index
        index.add(embeddings_array)
        print(f"✅ Added {index.ntotal} vectors to FAISS index")

        # Save FAISS index
        einstein_dir = project_root / ".einstein"
        einstein_dir.mkdir(exist_ok=True)

        faiss_path = einstein_dir / "embeddings.index"
        print(f"💾 Saving FAISS index to {faiss_path}")
        faiss.write_index(index, str(faiss_path))

        # Save metadata
        import json

        metadata_path = einstein_dir / "faiss_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(
                {
                    "version": "1.0",
                    "dimension": dimension,
                    "total_vectors": index.ntotal,
                    "files": file_metadata,
                    "created_at": time.time(),
                },
                f,
                indent=2,
            )

        print(f"💾 Saved metadata to {metadata_path}")

        # Test the index
        print("🧪 Testing FAISS index...")
        test_query = "wheel strategy"
        test_embedding, _ = await einstein._safe_get_query_embedding(test_query)

        if test_embedding is not None:
            test_embedding = test_embedding.reshape(1, -1)
            scores, indices = index.search(test_embedding, min(5, index.ntotal))
            print(f"   Test search returned {len(indices[0])} results")

            for i, (score, idx) in enumerate(zip(scores[0], indices[0], strict=False)):
                if idx >= 0 and idx < len(file_metadata):
                    meta = file_metadata[idx]
                    print(f"   {i+1}. {meta['file_path']} (score: {score:.3f})")

        print("✅ FAISS index built successfully!")
        print("📊 Final stats:")
        print(f"   - Index size: {index.ntotal} vectors")
        print(f"   - Dimension: {dimension}")
        print(f"   - Index file: {faiss_path}")
        print(f"   - Metadata file: {metadata_path}")

        return True

    except Exception as e:
        print(f"❌ Failed to build FAISS index: {e}")
        logger.error(f"FAISS index building failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = asyncio.run(build_faiss_index())
    sys.exit(0 if success else 1)
