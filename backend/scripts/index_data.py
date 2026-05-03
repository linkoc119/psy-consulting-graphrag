"""
Data Indexing Script
Run this after chunking to populate Qdrant and Neo4j
"""
import asyncio
import sys
import os
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from app.services.graph_indexer import index_all_chunks
from chunking.chunk_processor import process_all_documents, DEFAULT_DOCS_CONFIG
async def main():
    """Main indexing pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Index document chunks into Qdrant and Neo4j')
    parser.add_argument('--chunks-file', type=str, 
                       default=str(backend_dir / 'data' / 'processed_chunks.jsonl'),
                       help='Path to chunks JSONL file')
    parser.add_argument('--use-llm', action='store_true',
                       help='Use LLM for entity extraction (slower but more accurate)')
    parser.add_argument('--rechunk', action='store_true',
                       help='Re-run chunking before indexing')
    parser.add_argument('--clear-db', action='store_true',
                       help='Clear existing data before indexing (CAUTION)')
    
    args = parser.parse_args()
    
    # Step 0: Re-chunk if requested
    if args.rechunk:
        print("=" * 50)
        print("🔄 Re-chunking documents...")
        print("=" * 50)
        from chunking.chunk_processor import process_all_documents, save_chunks_to_jsonl
        
        chunks = process_all_documents(DEFAULT_DOCS_CONFIG)
        output_dir = backend_dir / 'data'
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / 'processed_chunks.jsonl'
        save_chunks_to_jsonl(chunks, str(output_file))
        print(f"✅ Chunking complete: {len(chunks)} chunks")
    
    # Step 1: Check chunks file exists
    chunks_file = Path(args.chunks_file)
    if not chunks_file.exists():
        print(f"❌ Chunks file not found: {chunks_file}")
        print("Run chunking first: python -m chunking.chunk_processor")
        return
    
    # Step 2: Clear database if requested
    if args.clear_db:
        print("=" * 50)
        print("⚠️  WARNING: Clearing all database data...")
        print("=" * 50)
        confirm = os.getenv('CONFIRM_CLEAR_DB', 'YES')
        if confirm != 'YES':
            print("Cancelled.")
            return
        
        try:
            from app.services.neo4j_service import get_neo4j_service
            neo4j = await get_neo4j_service()
            await neo4j.clear_all()
            print("✅ Neo4j cleared")
            
            from app.services.qdrant_service import get_qdrant_service
            qdrant = get_qdrant_service()
            qdrant.client.delete(collection_name=qdrant.collection_name)
            qdrant.create_collection(force=True)
            print("✅ Qdrant cleared")
            
        except Exception as e:
            print(f"❌ Error clearing databases: {e}")
            return
    
    # Step 3: Run indexing
    print("=" * 50)
    print("🔍 Starting Graph Indexing...")
    print(f"   Chunks file: {chunks_file}")
    print(f"   Use LLM: {args.use_llm}")
    print("=" * 50)
    
    try:
        stats = await index_all_chunks(
            chunks_file=str(chunks_file),
            use_llm=args.use_llm
        )
        
        print("\n" + "=" * 50)
        print("✅ INDEXING COMPLETE")
        print("=" * 50)
        print(f"Total chunks processed: {stats['total_chunks']}")
        print(f"Chunks indexed: {stats['chunks_indexed']}")
        print(f"Entities created: {stats['entities_created']}")
        print(f"Relationships created: {stats['relationships_created']}")
        print(f"Vectors uploaded: {stats['vectors_uploaded']}")
        print("\nNext steps:")
        print("  1. Start backend: uvicorn app.main:app --reload")
        print("  2. Open UI: http://localhost:3000")
        print("  3. Test chat: curl http://localhost:8000/chat/completion -X POST -H 'Content-Type: application/json' -d '{\"message\":\"Xin chào\"}'")
        
    except Exception as e:
        print(f"\n❌ INDEXING FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())