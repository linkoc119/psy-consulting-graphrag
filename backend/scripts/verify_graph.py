"""
Verify Graph Data
Check that nodes and relationships are properly indexed
"""
import asyncio
import sys
from pathlib import Path

backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from app.services.neo4j_service import get_neo4j_service

async def main():
    """Verify graph data"""
    print("=" * 60)
    print("🔍 VERIFYING GRAPH DATABASE")
    print("=" * 60)
    
    neo4j = await get_neo4j_service()
    
    # Count nodes by label
    labels = [
        "BenhLy", "TrieuChung", "DauHieuNguyHiem",
        "HanhDongPFA", "KyNangTuVan", "BuocTuVan",
        "Thuoc", "DoiTuong", "DocumentChunk"
    ]
    
    print("\n📊 Node Counts by Label:")
    print("-" * 60)
    total_nodes = 0
    for label in labels:
        try:
            count = await neo4j.count_nodes_by_label(label)
            total_nodes += count
            print(f"  {label:25s}: {count:6d}")
        except Exception as e:
            print(f"  {label:25s}: ERROR - {e}")
    
    print(f"\n  {'TOTAL':25s}: {total_nodes:6d}")
    
    # Count relationships by type
    print("\n🔗 Relationship Counts by Type:")
    print("-" * 60)
    rel_types = [
        "CO_TRIEU_CHUNG", "BAO_HIEU_NGUY_HIEM", "YEU_CAU_HANH_DONG",
        "DIEU_TRI_BANG", "QUAN_LY_BANG", "AP_DUNG_CHO",
        "BAO_GOM_BUOC", "NAM_TRONG_CHUNK"
    ]
    
    total_rels = 0
    for rel_type in rel_types:
        try:
            query = f"MATCH ()-[r:{rel_type}]->() RETURN count(r) as count"
            async with neo4j.driver.session() as session:
                result = await session.run(query)
                record = await result.single()
                count = record["count"] if record else 0
                total_rels += count
                print(f"  {rel_type:25s}: {count:6d}")
        except Exception as e:
            print(f"  {rel_type:25s}: ERROR - {e}")
    
    print(f"\n  {'TOTAL':25s}: {total_rels:6d}")
    
    # Sample queries
    print("\n📋 Sample Data:")
    print("-" * 60)
    
    # Show 5 BenhLy nodes
    print("\nTop 5 BenhLy nodes:")
    try:
        query = "MATCH (n:BenhLy) RETURN n.name, n.severity_level LIMIT 5"
        async with neo4j.driver.session() as session:
            result = await session.run(query)
            async for record in result:
                name = record["n.name"]
                sev = record.get("n.severity_level", "N/A")
                print(f"  - {name} (severity: {sev})")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Show 5 DocumentChunk nodes
    print("\nTop 5 DocumentChunk nodes:")
    try:
        query = "MATCH (n:DocumentChunk) RETURN n.id, n.doc_type, n.source LIMIT 5"
        async with neo4j.driver.session() as session:
            result = await session.run(query)
            async for record in result:
                print(f"  - {record['n.id'][:40]}... ({record['n.doc_type']})")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Graph connectivity check
    print("\n🔍 Connectivity Check:")
    print("-" * 60)
    
    # Check if any DocumentChunk is linked to entities
    try:
        query = """
        MATCH (dc:DocumentChunk)-[:NAM_TRONG_CHUNK]->(e)
        RETURN count(DISTINCT dc) as linked_chunks, count(DISTINCT e) as linked_entities
        """
        async with neo4j.driver.session() as session:
            result = await session.run(query)
            record = await result.single()
            linked_chunks = record["linked_chunks"]
            linked_entities = record["linked_entities"]
            print(f"  Chunks linked to entities: {linked_chunks}")
            print(f"  Unique entities linked: {linked_entities}")
    except Exception as e:
        print(f"  Error: {e}")
    
    await neo4j.close()
    
    print("\n" + "=" * 60)
    print("✅ Verification complete")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())