"""
Neo4j Graph Database Service for Knowledge Graph
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from neo4j import AsyncGraphDatabase, AsyncSession
from neo4j.exceptions import ServiceUnavailable
from config import settings

NEO4J_URI = settings.NEO4J_URI
NEO4J_USER = settings.NEO4J_USER
NEO4J_PASSWORD = settings.NEO4J_PASSWORD

logger = logging.getLogger(__name__)

class Neo4jService:
    """Service for interacting with Neo4j graph database"""
    
    def __init__(self, uri: str = NEO4J_URI, user: str = NEO4J_USER, password: str = NEO4J_PASSWORD):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
        
    async def connect(self):
        """Connect to Neo4j database"""
        try:
            self.driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
            # Verify connection
            async with self.driver.session() as session:
                await session.run("RETURN 1")
            logger.info(f"✅ Connected to Neo4j at {self.uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    async def close(self):
        """Close the driver connection"""
        if self.driver:
            await self.driver.close()
            logger.info("Neo4j connection closed")
    
    async def create_constraints(self):
        """Create uniqueness constraints for node labels"""
        constraints = [
            "CREATE CONSTRAINT benhly_id IF NOT EXISTS FOR (b:BenhLy) REQUIRE b.id IS UNIQUE",
            "CREATE CONSTRAINT trieuchung_id IF NOT EXISTS FOR (t:TrieuChung) REQUIRE t.id IS UNIQUE",
            "CREATE CONSTRAINT thuoc_id IF NOT EXISTS FOR (th:Thuoc) REQUIRE th.id IS UNIQUE",
            "CREATE CONSTRAINT kynang_id IF NOT EXISTS FOR (k:KyNangTuVan) REQUIRE k.id IS UNIQUE",
            "CREATE CONSTRAINT buoc_id IF NOT EXISTS FOR (b:BuocTuVan) REQUIRE b.id IS UNIQUE",
            "CREATE CONSTRAINT hanhdongpfa_id IF NOT EXISTS FOR (h:HanhDongPFA) REQUIRE h.id IS UNIQUE",
            "CREATE CONSTRAINT dautuong_id IF NOT EXISTS FOR (d:DauHieuNguyHiem) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT doituong_id IF NOT EXISTS FOR (d:DoiTuong) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT documentchunk_id IF NOT EXISTS FOR (d:DocumentChunk) REQUIRE d.id IS UNIQUE"
        ]
        
        async with self.driver.session() as session:
            for constraint in constraints:
                try:
                    await session.run(constraint)
                    logger.info(f"Created constraint: {constraint.split('FOR')[0].strip()}")
                except Exception as e:
                    logger.debug(f"Constraint may already exist: {e}")
    
    async def create_node(
        self,
        label: str,
        properties: Dict[str, Any],
        node_id: Optional[str] = None
    ) -> str:
        """Create a node with given label and properties"""
        if node_id is None:
            import uuid
            node_id = str(uuid.uuid4())

        # Merge id into properties for Cypher compatibility
        all_properties = {"id": node_id, **properties}

        query = f"""
        CREATE (n:{label} $props)
        RETURN n.id as id
        """

        async with self.driver.session() as session:
            result = await session.run(
                query,
                props=all_properties
            )
            record = await result.single()
            return record["id"]
    
    async def create_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Create a relationship between two nodes"""
        if properties is None:
            properties = {}
        
        query = f"""
        MATCH (a {{id: $source_id}}), (b {{id: $target_id}})
        CREATE (a)-[r:{rel_type}]->(b)
        SET r = $properties
        RETURN count(r) as created
        """
        
        async with self.driver.session() as session:
            result = await session.run(
                query,
                source_id=source_id,
                target_id=target_id,
                properties=properties
            )
            record = await result.single()
            return record["created"] > 0
    
    async def search_by_embedding(
        self,
        query_embedding: List[float],
        limit: int = 10,
        node_labels: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Search nodes by embedding similarity (requires vector index)
        This is a simplified version - in practice you'd need to create
        vector indexes on nodes with embeddings
        """
        # For now, we'll use a different approach: retrieve relevant nodes
        # from graph traversal initiated by vector search results
        
        # This would be implemented in rag_service where we:
        # 1. Get initial node IDs from Qdrant
        # 2. Expand via graph traversal from those nodes
        
        return []
    
    async def get_subgraph(
        self,
        node_ids: List[str],
        depth: int = 2,
        relationship_types: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Retrieve subgraph around given node IDs
        
        Args:
            node_ids: Starting node IDs
            depth: How many hops to traverse
            relationship_types: Filter specific relationship types
            
        Returns:
            List of nodes and relationships in subgraph
        """
        rel_filter = ""
        if relationship_types:
            rel_list = "|".join(relationship_types)
            rel_filter = f":{rel_list}"
        
        query = f"""
        MATCH (n)
        WHERE n.id IN $node_ids
        CALL apoc.path.subgraphAll(n, {{
            maxLevel: $depth,
            relationshipFilter: '{rel_filter}' if '{rel_filter}' != '' else null,
            bfs: true
        }})
        YIELD nodes, relationships
        RETURN nodes, relationships
        """
        
        async with self.driver.session() as session:
            result = await session.run(
                query,
                node_ids=node_ids,
                depth=depth
            )
            record = await result.single()
            
            if not record:
                return {"nodes": [], "relationships": []}
            
            nodes = []
            for node in record["nodes"]:
                nodes.append(dict(node))
            
            relationships = []
            for rel in record["relationships"]:
                relationships.append({
                    "type": rel.type,
                    "start": rel.start_node["id"],
                    "end": rel.end_node["id"],
                    "properties": dict(rel)
                })
            
            return {"nodes": nodes, "relationships": relationships}
    
    async def find_related_nodes(
        self,
        node_id: str,
        rel_type: Optional[str] = None,
        target_label: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict]:
        """Find nodes connected to given node"""
        query = """
        MATCH (n {id: $node_id})
        """
        
        if rel_type and target_label:
            query += f"MATCH (n)-[r:{rel_type}]->(m:{target_label}) "
        elif rel_type:
            query += f"MATCH (n)-[r:{rel_type}]->(m) "
        elif target_label:
            query += "MATCH (n)-[r]->(m) WHERE m:`" + target_label + "` "
        else:
            query += "MATCH (n)-[r]->(m) "
        
        query += """
        RETURN m as node, type(r) as rel_type, r as rel_properties
        LIMIT $limit
        """
        
        async with self.driver.session() as session:
            result = await session.run(
                query,
                node_id=node_id,
                limit=limit
            )
            
            nodes = []
            async for record in result:
                node_data = dict(record["node"])
                node_data["rel_type"] = record["rel_type"]
                node_data["rel_properties"] = dict(record["rel_properties"])
                nodes.append(node_data)
            
            return nodes
    
    async def get_node_by_name(
        self,
        name: str,
        label: str,
        aliases_field: str = "aliases"
    ) -> Optional[Dict]:
        """Find node by name or alias"""
        query = f"""
        MATCH (n:{label})
        WHERE n.name = $name OR (n.{aliases_field} AND $name IN n.{aliases_field})
        RETURN n
        LIMIT 1
        """
        
        async with self.driver.session() as session:
            result = await session.run(query, name=name)
            record = await result.single()
            return dict(record["n"]) if record else None
    
    async def count_nodes_by_label(self, label: str) -> int:
        """Count nodes with a specific label"""
        query = f"MATCH (n:{label}) RETURN count(n) as count"
        
        async with self.driver.session() as session:
            result = await session.run(query)
            record = await result.single()
            return record["count"]
    
    async def clear_all(self):
        """Clear all data (CAUTION: destructive)"""
        query = "MATCH (n) DETACH DELETE n"
        
        async with self.driver.session() as session:
            await session.run(query)
            logger.warning("All nodes and relationships deleted")


# Singleton instance
_neo4j_instance: Optional[Neo4jService] = None

async def get_neo4j_service() -> Neo4jService:
    """Get or create singleton Neo4j service instance"""
    global _neo4j_instance
    if _neo4j_instance is None:
        _neo4j_instance = Neo4jService()
        await _neo4j_instance.connect()
    return _neo4j_instance


if __name__ == "__main__":
    # Test
    import asyncio
    
    async def test():
        neo4j = await get_neo4j_service()
        await neo4j.create_constraints()
        
        # Count nodes
        count = await neo4j.count_nodes_by_label("BenhLy")
        print(f"BenhLy nodes count: {count}")
        
        await neo4j.close()
    
    asyncio.run(test())