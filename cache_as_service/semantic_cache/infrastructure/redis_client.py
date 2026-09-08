"""Redis client factory for connection management.

Handles Standalone, Sentinel, and Cluster topologies.
"""

import logging
from typing import List, Tuple, cast

import redis
from redis.cluster import RedisCluster
from redis.exceptions import RedisError
from redis.sentinel import Sentinel

from semantic_cache.core.config import SemanticCacheConfig
from semantic_cache.core.exceptions import RedisConnectionError

logger = logging.getLogger(__name__)


class RedisClientFactory:
    """Factory for creating Redis clients based on system configuration."""

    @staticmethod
    def create_client(config: SemanticCacheConfig) -> redis.Redis:
        """Creates and returns the appropriate Redis client.

        Evaluates the config to decide between Cluster, Sentinel, or Standalone mode.
        Validates the connection via an immediate `ping()`.

        Args:
            config: The SemanticCacheConfig containing host, port, and topology rules.

        Returns:
            An active, tested Redis client instance connected to the specified backend.

        Raises:
            RedisConnectionError: If the connection cannot be established or ping fails.
        """
        try:
            if config.redis_is_cluster:
                return RedisClientFactory._create_cluster_client(config)
            elif config.redis_sentinels and config.redis_sentinel_master:
                return RedisClientFactory._create_sentinel_client(config)
            else:
                return RedisClientFactory._create_standalone_client(config)
        except RedisError as e:
            logger.error("Failed to connect to Redis backend: %s", e)
            raise RedisConnectionError(f"Redis connection failed: {e}") from e

    @staticmethod
    def _create_standalone_client(config: SemanticCacheConfig) -> redis.Redis:
        logger.info("Connecting to standalone Redis at %s:%s", config.redis_host, config.redis_port)
        password = config.redis_password.get_secret_value() if config.redis_password else None

        client = redis.Redis(
            host=config.redis_host,
            port=config.redis_port,
            password=password,
            decode_responses=True
        )
        client.ping()
        return client

    @staticmethod
    def _create_sentinel_client(config: SemanticCacheConfig) -> redis.Redis:
        logger.info("Connecting to Redis Sentinel. Sentinels: %s", config.redis_sentinels)
        sentinel_nodes: List[Tuple[str, int]] = []

        if config.redis_sentinels:
            for node in config.redis_sentinels:
                host, port = node.split(":")
                sentinel_nodes.append((host, int(port)))

        password = config.redis_password.get_secret_value() if config.redis_password else None

        sentinel = Sentinel(
            sentinel_nodes,
            sentinel_kwargs={"password": password} if password else None
        )

        master_name = config.redis_sentinel_master if config.redis_sentinel_master else "mymaster"
        master = sentinel.master_for(
            master_name,
            password=password,
            decode_responses=True
        )
        master.ping()
        return master

    @staticmethod
    def _create_cluster_client(config: SemanticCacheConfig) -> redis.Redis:
        logger.info("Connecting to Redis Cluster at %s:%s", config.redis_host, config.redis_port)
        password = config.redis_password.get_secret_value() if config.redis_password else None

        client = RedisCluster(
            host=config.redis_host,
            port=config.redis_port,
            password=password,
            decode_responses=True
        )
        client.ping()
        return cast(redis.Redis, client)
