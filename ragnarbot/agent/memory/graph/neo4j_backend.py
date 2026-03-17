"""Neo4j graph backend for entity relationships."""

from __future__ import annotations

from loguru import logger


class Neo4jBackend:
    """Neo4j graph store for entities and relationships."""

    def __init__(self, uri: str = "bolt://localhost:7687", auth: tuple = ("neo4j", "ragnarbot")):
        self.uri = uri
        self.auth = auth
        self._driver = None

    async def initialize(self) -> None:
        """Initialize Neo4j driver and create constraints."""
        from neo4j import AsyncGraphDatabase
        self._driver = AsyncGraphDatabase.driver(self.uri, auth=self.auth)

        async with self._driver.session() as session:
            # Unique constraint on entity name
            await session.run(
                "CREATE CONSTRAINT entity_name IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE e.name IS UNIQUE"
            )
            # Index on entity type
            await session.run(
                "CREATE INDEX entity_type IF NOT EXISTS "
                "FOR (e:Entity) ON (e.entity_type)"
            )
        logger.info(f"Neo4j backend initialized at {self.uri}")

    async def upsert_entity(
        self,
        name: str,
        entity_type: str = "",
        aliases: list[str] | None = None,
    ) -> str:
        """Create or update an entity. Returns entity name."""
        async with self._driver.session() as session:
            result = await session.run(
                """
                MERGE (e:Entity {name: $name})
                ON CREATE SET
                    e.entity_type = $entity_type,
                    e.aliases = $aliases,
                    e.created_at = datetime()
                ON MATCH SET
                    e.entity_type = CASE WHEN $entity_type <> '' THEN $entity_type ELSE e.entity_type END,
                    e.aliases = CASE WHEN size($aliases) > 0
                        THEN apoc.coll.union(coalesce(e.aliases, []), $aliases)
                        ELSE e.aliases END
                RETURN e.name AS name
                """,
                name=name,
                entity_type=entity_type,
                aliases=aliases or [],
            )
            record = await result.single()
            return record["name"] if record else name

    async def upsert_relation(
        self,
        subject: str,
        predicate: str,
        object_name: str,
        fact_id: str,
        confidence: float = 1.0,
    ) -> None:
        """Create or update a relationship between entities."""
        async with self._driver.session() as session:
            await session.run(
                """
                MERGE (s:Entity {name: $subject})
                MERGE (o:Entity {name: $object})
                MERGE (s)-[r:RELATION {type: $predicate}]->(o)
                ON CREATE SET
                    r.fact_id = $fact_id,
                    r.confidence = $confidence,
                    r.created_at = datetime()
                ON MATCH SET
                    r.fact_id = $fact_id,
                    r.confidence = $confidence,
                    r.updated_at = datetime()
                """,
                subject=subject,
                predicate=predicate,
                object=object_name,
                fact_id=fact_id,
                confidence=confidence,
            )

    async def invalidate_relation(self, fact_id: str) -> None:
        """Mark a relationship as invalid (set valid_until)."""
        async with self._driver.session() as session:
            await session.run(
                """
                MATCH ()-[r:RELATION {fact_id: $fact_id}]->()
                SET r.valid_until = datetime()
                """,
                fact_id=fact_id,
            )

    async def get_related_facts(
        self,
        entity_name: str,
        depth: int = 2,
        limit: int = 20,
    ) -> list[dict]:
        """Get facts related to an entity via graph traversal."""
        async with self._driver.session() as session:
            result = await session.run(
                """
                MATCH (start:Entity {name: $name})
                MATCH path = (start)-[r:RELATION*1..""" + str(depth) + """]->(target:Entity)
                WHERE ALL(rel IN r WHERE rel.valid_until IS NULL)
                WITH target, r, length(path) AS dist
                UNWIND r AS rel
                RETURN DISTINCT
                    startNode(rel).name AS subject,
                    rel.type AS predicate,
                    endNode(rel).name AS object,
                    rel.fact_id AS fact_id,
                    rel.confidence AS confidence,
                    dist AS depth
                ORDER BY dist ASC, rel.confidence DESC
                LIMIT $limit
                """,
                name=entity_name,
                limit=limit,
            )
            records = [record.data() async for record in result]
            return records

    async def resolve_entity_alias(self, name: str) -> str:
        """Resolve an entity name through aliases."""
        async with self._driver.session() as session:
            result = await session.run(
                """
                MATCH (e:Entity)
                WHERE e.name = $name OR $name IN e.aliases
                RETURN e.name AS canonical_name
                LIMIT 1
                """,
                name=name,
            )
            record = await result.single()
            return record["canonical_name"] if record else name

    async def resolve_entity_from_query(self, query: str) -> str | None:
        """Try to find an entity mentioned in a query text."""
        # Simple approach: check if any entity name appears in the query
        async with self._driver.session() as session:
            result = await session.run(
                """
                MATCH (e:Entity)
                WHERE toLower($search_text) CONTAINS toLower(e.name)
                   OR ANY(alias IN e.aliases WHERE toLower($search_text) CONTAINS toLower(alias))
                RETURN e.name AS name
                ORDER BY size(e.name) DESC
                LIMIT 1
                """,
                search_text=query,
            )
            record = await result.single()
            return record["name"] if record else None

    async def get_entity_neighbors(self, entity_name: str) -> list[dict]:
        """Get direct neighbors of an entity."""
        async with self._driver.session() as session:
            result = await session.run(
                """
                MATCH (e:Entity {name: $name})-[r:RELATION]-(other:Entity)
                WHERE r.valid_until IS NULL
                RETURN
                    other.name AS name,
                    other.entity_type AS entity_type,
                    r.type AS relation,
                    r.confidence AS confidence,
                    startNode(r).name = $name AS outgoing
                """,
                name=entity_name,
            )
            return [record.data() async for record in result]

    async def purge_all(self) -> dict[str, int]:
        """Delete ALL nodes and relationships. Returns counts."""
        if not self._driver:
            return {"nodes": 0, "relationships": 0}
        async with self._driver.session() as session:
            result = await session.run("MATCH (n) DETACH DELETE n RETURN count(n) AS cnt")
            record = await result.single()
            return {"nodes": record["cnt"] if record else 0}

    async def close(self) -> None:
        if self._driver:
            await self._driver.close()
