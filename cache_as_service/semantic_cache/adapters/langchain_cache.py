"""LangChain adapter for the Semantic Cache.

Allows dropping the Semantic Cache into any LangChain LLM or ChatModel via:
`from langchain.globals import set_llm_cache`
`set_llm_cache(LangChainSemanticCache(cache_manager))`
"""

import logging
from typing import Any, Optional, Sequence

from langchain_core.caches import BaseCache
from langchain_core.outputs import Generation

from semantic_cache.core.cache_manager import SemanticCacheManager

logger = logging.getLogger(__name__)


class LangChainSemanticCache(BaseCache):
    """LangChain BaseCache implementation backed by our SemanticCacheManager."""

    def __init__(self, cache_manager: SemanticCacheManager):
        """Initialize the LangChain adapter.

        Args:
            cache_manager: The initialized core SemanticCacheManager.
        """
        self.cache_manager = cache_manager

    def lookup(self, prompt: str, llm_string: str) -> Optional[Sequence[Generation]]:
        """Look up based on prompt and llm_string.

        Args:
            prompt: The string prompt provided to the LLM.
            llm_string: The stringified LLM config (used as scope/metadata here).

        Returns:
            A Sequence of Generations if a hit occurs, else None.
        """
        try:
            # We pass llm_string in metadata to track which LLM generated it,
            # though semantic search primarily relies on the prompt.
            result = self.cache_manager.search(prompt, llm_string=llm_string)
            
            if result:
                logger.info(f"LangChain Cache HIT for prompt: {prompt[:30]}... (Similarity: {result['similarity']:.4f})")
                
                # Reconstruct the LangChain Generation object
                return [Generation(text=result["response"])]
                
            return None
            
        except Exception as e:
            # LangChain caches should generally fail silently and fallback to actual LLM call
            logger.error(f"LangChain semantic cache lookup failed: {e}")
            return None

    def update(self, prompt: str, llm_string: str, return_val: Sequence[Generation]) -> None:
        """Update cache based on prompt and llm_string.

        Args:
            prompt: The string prompt provided to the LLM.
            llm_string: The stringified LLM config.
            return_val: The sequence of Generations to cache.
        """
        if not return_val:
            return

        try:
            # Extract the raw text from the LLM completion
            text_response = return_val[0].text
            
            # Check if metadata in llm_string suggests keeping forever
            # (In standard LangChain this isn't native, but if you inject tags manually you can parse them)
            keep_forever = "KEEP_FOREVER" in llm_string
            
            # Save to Redis
            self.cache_manager.set(
                query=prompt,
                response=text_response,
                metadata={"llm_string": llm_string, "source": "langchain"},
                keep_forever=keep_forever
            )
            logger.debug("LangChain Cache UPDATED.")
            
        except Exception as e:
            # Fail silently to not crash the main LLM pipeline
            logger.error(f"LangChain semantic cache update failed: {e}")

    def clear(self, **kwargs: Any) -> None:
        """Clear cache that can take additional keyword arguments."""
        try:
            self.cache_manager.purge()
        except Exception as e:
            logger.error(f"LangChain semantic cache clear failed: {e}")
