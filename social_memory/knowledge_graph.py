# -*- coding: utf-8 -*-
"""Knowledge Graph Manager for Social Memory System"""

import logging
logger = logging.getLogger(__name__)

import json
from typing import Optional, List, Dict, Any
from datetime import datetime

try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None
    logger.warning("Warning: neo4j not installed. Install with: pip install neo4j")

from .config import GraphStoreConfig
from .models import SocialUser, SocialPost, SocialComment, PlatformType


class KnowledgeGraphManager:
    """Manager for knowledge graph using Neo4j"""

    def __init__(self, config: GraphStoreConfig):
        """Initialize knowledge graph manager

        Args:
            config: Graph store configuration
        """
        self.config = config
        self.driver: Optional[GraphDatabase.driver] = None
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize Neo4j connection

        Returns:
            bool: True if initialization successful
        """
        try:
            if GraphDatabase is None:
                logger.warning("Error: neo4j not installed")
                return False

            # Create driver with connection pool settings
            self.driver = GraphDatabase.driver(
                self.config.uri,
                auth=(self.config.user, self.config.password),
                max_connection_lifetime=self.config.max_connection_lifetime,
                max_connection_pool_size=self.config.max_connection_pool_size,
                connection_acquisition_timeout=self.config.connection_acquisition_timeout,
            )

            # Test connection
            self.driver.verify_connectivity()

            # Create indexes and constraints
            await self._create_indexes()

            self._initialized = True
            logger.info("Knowledge graph initialized: %s", self.config.uri)
            return True

        except Exception as e:
            logger.error("Failed to initialize knowledge graph: %s", e)
            return False

    async def _create_indexes(self) -> None:
        """Create indexes and constraints for better query performance"""
        if not self.driver:
            return

        with self.driver.session(database=self.config.database) as session:
            # User constraints and indexes
            session.run("""
                CREATE CONSTRAINT user_id_unique IF NOT EXISTS
                FOR (u:User) REQUIRE u.user_id IS UNIQUE
            """)

            session.run("""
                CREATE INDEX user_username_idx IF NOT EXISTS
                FOR (u:User) ON (u.username)
            """)

            # Post constraints and indexes
            session.run("""
                CREATE CONSTRAINT post_id_unique IF NOT EXISTS
                FOR (p:Post) REQUIRE p.post_id IS UNIQUE
            """)

            # Comment constraints and indexes
            session.run("""
                CREATE CONSTRAINT comment_id_unique IF NOT EXISTS
                FOR (c:Comment) REQUIRE c.comment_id IS UNIQUE
            """)

            # Topic indexes
            session.run("""
                CREATE INDEX topic_name_idx IF NOT EXISTS
                FOR (t:Topic) ON (t.name)
            """)

            # Submolt constraints and indexes
            session.run("""
                CREATE CONSTRAINT submolt_name_unique IF NOT EXISTS
                FOR (s:Submolt) REQUIRE s.name IS UNIQUE
            """)

            logger.info("Knowledge graph indexes created")

    def _execute_query(
        self, query: str, parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute a Cypher query

        Args:
            query: Cypher query string
            parameters: Query parameters

        Returns:
            List of result records
        """
        if not self.driver:
            return []

        try:
            with self.driver.session(database=self.config.database) as session:
                result = session.run(query, parameters)
                return [record.data() for record in result]
        except Exception as e:
            logger.error("Error executing query: %s", e)
            return []

    async def add_user_node(self, user: SocialUser) -> bool:
        """Add or update a user node in the graph

        Args:
            user: Social user object

        Returns:
            bool: True if operation successful
        """
        try:
            query = """
                MERGE (u:User {user_id: $user_id})
                SET u.username = $username,
                    u.platform = $platform,
                    u.profile_url = $profile_url,
                    u.followers_count = $followers_count,
                    u.following_count = $following_count,
                    u.verified = $verified,
                    u.updated_at = datetime()
                RETURN u
            """

            self._execute_query(
                query,
                {
                    "user_id": user.user_id,
                    "username": user.username,
                    "platform": user.platform.value,
                    "profile_url": user.profile_url,
                    "followers_count": user.followers_count,
                    "following_count": user.following_count,
                    "verified": user.verified,
                },
            )

            return True

        except Exception as e:
            logger.error("Error adding user node: %s", e)
            return False

    async def add_post_node(self, post: SocialPost) -> bool:
        """Add or update a post node in the graph

        Args:
            post: Social post object

        Returns:
            bool: True if operation successful
        """
        try:
            # First ensure user exists
            await self.add_user_node(post.author)

            # Add post node and create POSTED relationship
            query = """
                MERGE (u:User {user_id: $user_id})
                MERGE (p:Post {post_id: $post_id})
                SET p.content = $content,
                    p.platform = $platform,
                    p.timestamp = datetime($timestamp),
                    p.sentiment = $sentiment,
                    p.upvotes = $upvotes,
                    p.downvotes = $downvotes,
                    p.comment_count = $comment_count,
                    p.updated_at = datetime()
                MERGE (u)-[r:POSTED]->(p)
                SET r.timestamp = datetime($timestamp)
                RETURN p
            """

            self._execute_query(
                query,
                {
                    "user_id": post.author.user_id,
                    "post_id": post.post_id,
                    "content": post.content,
                    "platform": post.platform.value,
                    "timestamp": post.timestamp.isoformat(),
                    "sentiment": post.sentiment.value if post.sentiment else None,
                    "upvotes": post.engagement.upvotes,
                    "downvotes": post.engagement.downvotes,
                    "comment_count": post.engagement.comment_count,
                },
            )

            # Add topic relationships if available
            if post.topics:
                for topic in post.topics:
                    await self._add_topic_to_post(post.post_id, topic)

            return True

        except Exception as e:
            logger.error("Error adding post node: %s", e)
            return False

    async def _add_topic_to_post(self, post_id: str, topic: str) -> bool:
        """Add topic relationship to a post

        Args:
            post_id: Post ID
            topic: Topic name

        Returns:
            bool: True if operation successful
        """
        try:
            query = """
                MERGE (p:Post {post_id: $post_id})
                MERGE (t:Topic {name: $topic})
                MERGE (p)-[r:ABOUT]->(t)
                RETURN p, t
            """

            self._execute_query(
                query,
                {
                    "post_id": post_id,
                    "topic": topic,
                },
            )

            return True

        except Exception as e:
            logger.error("Error adding topic to post: %s", e)
            return False

    async def add_comment_node(self, comment: SocialComment) -> bool:
        """Add or update a comment node in the graph

        Args:
            comment: Social comment object

        Returns:
            bool: True if operation successful
        """
        try:
            # First ensure user exists
            await self.add_user_node(comment.author)

            # Add comment node and create relationships
            query = """
                MERGE (u:User {user_id: $user_id})
                MERGE (p:Post {post_id: $post_id})
                MERGE (c:Comment {comment_id: $comment_id})
                SET c.content = $content,
                    c.platform = $platform,
                    c.timestamp = datetime($timestamp),
                    c.sentiment = $sentiment,
                    c.author_type = $author_type,
                    c.learning_value = $learning_value,
                    c.updated_at = datetime()
                MERGE (u)-[r:COMMENTED]->(c)
                SET r.timestamp = datetime($timestamp)
                MERGE (c)-[r2:ON]->(p)
                SET r2.timestamp = datetime($timestamp)
                RETURN c
            """

            self._execute_query(
                query,
                {
                    "user_id": comment.author.user_id,
                    "post_id": comment.target_post_id,
                    "comment_id": comment.comment_id,
                    "content": comment.content,
                    "platform": comment.platform.value,
                    "timestamp": comment.timestamp.isoformat(),
                    "sentiment": comment.sentiment.value if comment.sentiment else None,
                    "author_type": comment.author_type.value,
                    "learning_value": comment.learning_value,
                },
            )

            # Add reply relationship if applicable
            if comment.reply_to_comment_id:
                await self._add_reply_relationship(
                    comment.comment_id, comment.reply_to_comment_id
                )

            return True

        except Exception as e:
            logger.error("Error adding comment node: %s", e)
            return False

    async def _add_reply_relationship(
        self, comment_id: str, reply_to_comment_id: str
    ) -> bool:
        """Add reply relationship between comments

        Args:
            comment_id: Comment ID
            reply_to_comment_id: Parent comment ID

        Returns:
            bool: True if operation successful
        """
        try:
            query = """
                MATCH (c1:Comment {comment_id: $comment_id})
                MATCH (c2:Comment {comment_id: $reply_to_comment_id})
                MERGE (c1)-[r:REPLIED_TO]->(c2)
                RETURN c1, c2
            """

            self._execute_query(
                query,
                {
                    "comment_id": comment_id,
                    "reply_to_comment_id": reply_to_comment_id,
                },
            )

            return True

        except Exception as e:
            logger.error("Error adding reply relationship: %s", e)
            return False

    async def create_interaction_relation(
        self,
        from_user_id: str,
        to_user_id: Optional[str],
        interaction_type: str,
        target_post_id: Optional[str],
        timestamp: datetime,
    ) -> bool:
        """Create an interaction relationship between users

        Args:
            from_user_id: Source user ID
            to_user_id: Target user ID
            interaction_type: Type of interaction
            target_post_id: Optional target post ID
            timestamp: Interaction timestamp

        Returns:
            bool: True if operation successful
        """
        try:
            query = """
                MATCH (u1:User {user_id: $from_user_id})
                """

            parameters = {
                "from_user_id": from_user_id,
                "interaction_type": interaction_type,
                "timestamp": timestamp.isoformat(),
            }

            if to_user_id:
                query += """
                    MATCH (u2:User {user_id: $to_user_id})
                    MERGE (u1)-[r:INTERACTED_WITH]->(u2)
                    """
                parameters["to_user_id"] = to_user_id
            else:
                query += """
                    MERGE (u1)-[r:INTERACTED]->(null)
                    """

            query += """
                SET r.type = $interaction_type,
                    r.timestamp = datetime($timestamp)
                """

            if target_post_id:
                query += """
                    MATCH (p:Post {post_id: $target_post_id})
                    MERGE (u1)-[r2:TARGETED]->(p)
                    SET r2.timestamp = datetime($timestamp)
                    """
                parameters["target_post_id"] = target_post_id

            self._execute_query(query, parameters)

            return True

        except Exception as e:
            logger.error("Error creating interaction relation: %s", e)
            return False

    async def find_similar_users(
        self, user_id: str, limit: int = 10, min_common_topics: int = 1
    ) -> List[Dict[str, Any]]:
        """Find users with similar interests based on posts and comments

        Args:
            user_id: User ID to find similar users for
            limit: Maximum number of results
            min_common_topics: Minimum common topics required

        Returns:
            List of similar users with similarity scores
        """
        try:
            query = """
                MATCH (u1:User {user_id: $user_id})-[:POSTED|COMMENTED]->(:Post|Comment)-[:ABOUT]->(t:Topic)
                MATCH (u2:User)-[:POSTED|COMMENTED]->(:Post|Comment)-[:ABOUT]->(t)
                WHERE u1 <> u2
                WITH u2, count(DISTINCT t) as common_topics
                WHERE common_topics >= $min_common_topics
                RETURN u2.user_id as user_id,
                       u2.username as username,
                       u2.platform as platform,
                       u2.followers_count as followers_count,
                       common_topics
                ORDER BY common_topics DESC
                LIMIT $limit
            """

            results = self._execute_query(
                query,
                {
                    "user_id": user_id,
                    "min_common_topics": min_common_topics,
                    "limit": limit,
                },
            )

            return results

        except Exception as e:
            logger.error("Error finding similar users: %s", e)
            return []

    async def analyze_propagation_paths(
        self, post_id: str, max_depth: int = 3
    ) -> Dict[str, Any]:
        """Analyze content propagation paths from a post

        Args:
            post_id: Post ID to analyze
            max_depth: Maximum depth of propagation analysis

        Returns:
            Dictionary containing propagation analysis
        """
        try:
            query = """
                MATCH path = (p:Post {post_id: $post_id})<-[:ON]-(:Comment)<-[:REPLIED_TO*0..{}]-(c:Comment)
                WITH p, path, c
                MATCH (u:User)-[:COMMENTED]->(c)
                RETURN path,
                       u.user_id as user_id,
                       u.username as username,
                       c.comment_id as comment_id,
                       c.content as content,
                       length([n IN nodes(path) WHERE n:Comment]) as depth
                ORDER BY depth, c.timestamp
            """.format(max_depth)

            results = self._execute_query(
                query,
                {
                    "post_id": post_id,
                },
            )

            return {
                "post_id": post_id,
                "total_comments": len(results),
                "propagation_paths": results,
                "max_depth_reached": max([r["depth"] for r in results])
                if results
                else 0,
            }

        except Exception as e:
            logger.error("Error analyzing propagation paths: %s", e)
            return {}

    async def detect_communities(
        self, min_community_size: int = 3
    ) -> List[Dict[str, Any]]:
        """Detect user communities based on interaction patterns

        Args:
            min_community_size: Minimum size for a community

        Returns:
            List of detected communities
        """
        try:
            # This is a simplified community detection
            # In production, use proper graph algorithms like Label Propagation
            query = """
                MATCH (u1:User)-[:INTERACTED_WITH]->(u2:User)
                WITH u1, collect(DISTINCT u2.user_id) as connections
                WHERE size(connections) >= $min_community_size - 1
                RETURN u1.user_id as user_id,
                       u1.username as username,
                       u1.platform as platform,
                       connections,
                       size(connections) as connection_count
                ORDER BY connection_count DESC
            """

            results = self._execute_query(
                query,
                {
                    "min_community_size": min_community_size,
                },
            )

            return results

        except Exception as e:
            logger.error("Error detecting communities: %s", e)
            return []

    async def get_user_network(self, user_id: str, depth: int = 2) -> Dict[str, Any]:
        """Get a user's social network

        Args:
            user_id: User ID
            depth: Network depth to explore

        Returns:
            Dictionary containing network information
        """
        try:
            query = """
                MATCH (u:User {user_id: $user_id})
                OPTIONAL MATCH (u)-[r:INTERACTED_WITH*1..{}]-(other:User)
                RETURN u.user_id as center_user_id,
                       u.username as center_username,
                       collect(DISTINCT {
                           user_id: other.user_id,
                           username: other.username,
                           interaction_types: collect(DISTINCT type(r))
                       }) as network_users
            """.format(depth)

            results = self._execute_query(
                query,
                {
                    "user_id": user_id,
                },
            )

            if results:
                return {
                    "center_user_id": results[0]["center_user_id"],
                    "center_username": results[0]["center_username"],
                    "network_users": results[0]["network_users"],
                    "network_size": len(results[0]["network_users"]),
                    "depth": depth,
                }

            return {}

        except Exception as e:
            logger.error("Error getting user network: %s", e)
            return {}

    # ── Enhanced graph algorithms and Submolt support ──

    async def add_submolt_node(
        self, name: str, description: str = "", member_count: int = 0
    ) -> bool:
        """Add or update a Submolt (community) node

        Args:
            name: Submolt name
            description: Submolt description
            member_count: Number of members

        Returns:
            bool: True if successful
        """
        try:
            self._execute_query(
                """
                MERGE (s:Submolt {name: $name})
                SET s.description = $description,
                    s.member_count = $member_count,
                    s.updated_at = datetime()
                RETURN s
                """,
                {"name": name, "description": description, "member_count": member_count},
            )
            return True
        except Exception as e:
            logger.error("Error adding submolt node: %s", e)
            return False

    async def add_member_of_relation(self, user_id: str, submolt_name: str) -> bool:
        """Create MEMBER_OF relationship between user and submolt"""
        try:
            self._execute_query(
                """
                MATCH (u:User {user_id: $user_id})
                MERGE (s:Submolt {name: $submolt_name})
                MERGE (u)-[r:MEMBER_OF]->(s)
                SET r.joined_at = datetime()
                """,
                {"user_id": user_id, "submolt_name": submolt_name},
            )
            return True
        except Exception as e:
            logger.error("Error adding MEMBER_OF relation: %s", e)
            return False

    async def add_posted_in_relation(self, post_id: str, submolt_name: str) -> bool:
        """Create POSTED_IN relationship between post and submolt"""
        try:
            self._execute_query(
                """
                MATCH (p:Post {post_id: $post_id})
                MERGE (s:Submolt {name: $submolt_name})
                MERGE (p)-[r:POSTED_IN]->(s)
                """,
                {"post_id": post_id, "submolt_name": submolt_name},
            )
            return True
        except Exception as e:
            logger.error("Error adding POSTED_IN relation: %s", e)
            return False

    async def create_weighted_interaction(
        self,
        from_user_id: str,
        to_user_id: str,
        interaction_type: str,
        timestamp: datetime,
    ) -> bool:
        """Create or update a weighted INTERACTED_WITH relationship

        Weights: upvote=0.3, comment=0.7, reply=1.0
        """
        weight_map = {"upvote": 0.3, "comment": 0.7, "reply": 1.0}
        quality_weight = weight_map.get(interaction_type, 0.5)

        try:
            self._execute_query(
                """
                MATCH (u1:User {user_id: $from_id})
                MATCH (u2:User {user_id: $to_id})
                MERGE (u1)-[r:INTERACTED_WITH]->(u2)
                SET r.quality_weight = COALESCE(r.quality_weight, 0) + $weight,
                    r.interaction_count = COALESCE(r.interaction_count, 0) + 1,
                    r.last_interaction = datetime($timestamp),
                    r.type = $interaction_type
                """,
                {
                    "from_id": from_user_id,
                    "to_id": to_user_id,
                    "weight": quality_weight,
                    "timestamp": timestamp.isoformat(),
                    "interaction_type": interaction_type,
                },
            )
            return True
        except Exception as e:
            logger.error("Error creating weighted interaction: %s", e)
            return False

    async def compute_pagerank(
        self, submolt_name: Optional[str] = None, top_n: int = 20
    ) -> List[Dict[str, Any]]:
        """Compute simplified PageRank to identify influential users

        Uses INTERACTED_WITH relationship weights. If submolt_name is provided,
        limits to users who are members of that submolt.

        Returns:
            List of {user_id, username, influence_score} sorted by score
        """
        try:
            if submolt_name:
                query = """
                    MATCH (u:User)-[:MEMBER_OF]->(:Submolt {name: $submolt})
                    WITH collect(u) as users
                    UNWIND users as u
                    OPTIONAL MATCH (other:User)-[r:INTERACTED_WITH]->(u)
                    WHERE other IN users
                    WITH u, COALESCE(SUM(r.quality_weight), 0) as incoming_weight,
                         COALESCE(COUNT(r), 0) as incoming_count
                    RETURN u.user_id as user_id,
                           u.username as username,
                           incoming_weight + incoming_count * 0.1 as influence_score
                    ORDER BY influence_score DESC
                    LIMIT $top_n
                """
                return self._execute_query(
                    query, {"submolt": submolt_name, "top_n": top_n}
                )
            else:
                query = """
                    MATCH (u:User)
                    OPTIONAL MATCH (other:User)-[r:INTERACTED_WITH]->(u)
                    WITH u, COALESCE(SUM(r.quality_weight), 0) as incoming_weight,
                         COALESCE(COUNT(r), 0) as incoming_count
                    RETURN u.user_id as user_id,
                           u.username as username,
                           incoming_weight + incoming_count * 0.1 as influence_score
                    ORDER BY influence_score DESC
                    LIMIT $top_n
                """
                return self._execute_query(query, {"top_n": top_n})
        except Exception as e:
            logger.error("Error computing PageRank: %s", e)
            return []

    async def detect_communities_lpa(
        self, min_community_size: int = 3
    ) -> List[Dict[str, Any]]:
        """Detect communities using Label Propagation-like approach

        Groups users by their strongest interaction clusters.

        Returns:
            List of community dicts with members and topics
        """
        try:
            # Get all connected components via INTERACTED_WITH
            query = """
                MATCH (u1:User)-[r:INTERACTED_WITH]-(u2:User)
                WITH u1, u2, r.quality_weight as weight
                ORDER BY weight DESC
                WITH u1,
                     collect({user_id: u2.user_id, username: u2.username,
                              weight: weight}) as connections
                WHERE size(connections) >= $min_size - 1
                RETURN u1.user_id as center_user_id,
                       u1.username as center_username,
                       connections,
                       size(connections) as community_size
                ORDER BY community_size DESC
            """
            results = self._execute_query(
                query, {"min_size": min_community_size}
            )

            # Post-process into community clusters
            communities = []
            seen_users = set()
            for r in results:
                center = r["center_user_id"]
                if center in seen_users:
                    continue
                members = [
                    {"user_id": center, "username": r["center_username"]}
                ]
                for conn in r["connections"]:
                    if conn["user_id"] not in seen_users:
                        members.append(
                            {"user_id": conn["user_id"], "username": conn["username"]}
                        )
                        seen_users.add(conn["user_id"])
                seen_users.add(center)
                if len(members) >= min_community_size:
                    communities.append(
                        {"members": members, "size": len(members)}
                    )

            return communities
        except Exception as e:
            logger.error("Error detecting communities: %s", e)
            return []

    async def find_path_to_influencer(
        self, user_id: str, target_user_id: str, max_depth: int = 4
    ) -> List[Dict[str, Any]]:
        """Find shortest interaction path from user to an influencer

        Args:
            user_id: Starting user
            target_user_id: Target influencer
            max_depth: Maximum hops

        Returns:
            List of users along the path
        """
        try:
            query = """
                MATCH path = shortestPath(
                    (start:User {{user_id: $start_id}})-
                    [:INTERACTED_WITH*1..{depth}]-
                    (end:User {{user_id: $end_id}})
                )
                RETURN [n IN nodes(path) | {{
                    user_id: n.user_id,
                    username: n.username
                }}] as path_nodes,
                length(path) as hops
            """.format(depth=max_depth)

            results = self._execute_query(
                query, {"start_id": user_id, "end_id": target_user_id}
            )
            if results:
                return results[0].get("path_nodes", [])
            return []
        except Exception as e:
            logger.error("Error finding path to influencer: %s", e)
            return []

    async def close(self) -> None:
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            self.driver = None
        self._initialized = False
        logger.info("Knowledge graph connection closed")
