"""
SaxoBank Token Lifecycle Management Service

Handles:
- Token storage (per-user isolation)
- Automatic token refresh
- Cache fallback for offline/expired scenarios
- Multi-tenant security

Storage:
- data/users/{user_id}/saxobank/tokens.json
- data/users/{user_id}/saxobank/api_cache/positions_YYYYMMDD_HHMMSS.json
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

from api.services.user_fs import UserScopedFS
from connectors.saxo_api import SaxoOAuthClient

logger = logging.getLogger(__name__)


class SaxoAuthService:
    """
    Manages SaxoBank OAuth2 tokens and caching for a specific user.

    Multi-tenant aware: Each user has isolated token storage.
    """

    def __init__(self, user_id: str, project_root: str = "."):
        """
        Initialize auth service for a user.

        Args:
            user_id: User identifier (e.g., "demo", "jack")
            project_root: Project root directory (defaults to current dir)
        """
        self.user_id = user_id
        self.user_fs = UserScopedFS(project_root, user_id)
        self.oauth_client = SaxoOAuthClient()

        # Paths relative to user's saxobank directory
        self.tokens_path = "saxobank/tokens.json"
        self.cache_dir = "saxobank/api_cache"

        logger.debug(f"SaxoAuthService initialized for user '{user_id}'")

    def is_connected(self) -> bool:
        """
        Check if user has valid refresh token.

        Returns:
            True if user is connected (has refresh_token), False otherwise

        Note: Does NOT check if access_token is valid (use get_valid_access_token for that)
        """
        try:
            tokens = self._load_tokens()
            if not tokens:
                return False

            refresh_token = tokens.get("refresh_token")
            if not refresh_token:
                return False

            # Check refresh token expiration (24h for Self-Developer accounts)
            last_update = tokens.get("last_update")
            if last_update:
                last_update_dt = datetime.fromisoformat(last_update)
                age_hours = (datetime.now() - last_update_dt).total_seconds() / 3600

                if age_hours > 24:
                    logger.warning(f"⚠️ Refresh token expired (age: {age_hours:.1f}h > 24h)")
                    return False

            return True

        except Exception as e:
            logger.error(f"Error checking connection status: {e}")
            return False

    async def get_valid_access_token(self) -> Optional[str]:
        """
        Get a valid access token (auto-refresh if expired).

        Returns:
            Valid access_token string, or None if user must reconnect

        Flow:
            1. Load tokens from storage
            2. Check if access_token expired (< 2min remaining)
            3. If expired, refresh automatically
            4. If refresh fails, return None (user must re-authenticate)
        """
        try:
            tokens = self._load_tokens()
            if not tokens:
                logger.debug("No tokens found - user not connected")
                return None

            # Check expiration
            expires_at_str = tokens.get("expires_at")
            if not expires_at_str:
                logger.warning("Token missing 'expires_at' field")
                return None

            expires_at = datetime.fromisoformat(expires_at_str)
            now = datetime.now()

            # Refresh if < 2 minutes remaining
            if expires_at <= now + timedelta(minutes=2):
                logger.info(f"🔄 Access token expires soon, refreshing...")
                refresh_token = tokens.get("refresh_token")

                if not refresh_token:
                    logger.error("No refresh_token available")
                    return None

                # Refresh
                new_tokens = await self.oauth_client.refresh_access_token(refresh_token)
                await self.save_tokens(new_tokens)

                return new_tokens.get("access_token")

            # Token still valid
            logger.debug(f"✅ Access token valid until {expires_at.isoformat()}")
            return tokens.get("access_token")

        except Exception as e:
            logger.error(f"Error getting valid access token: {e}")
            return None

    async def save_tokens(self, tokens: Dict[str, Any]) -> None:
        """
        Save OAuth2 tokens to user-scoped storage.

        Args:
            tokens: Token response from Saxo
                {
                    "access_token": "...",
                    "refresh_token": "...",
                    "expires_in": 1200,
                    "token_type": "Bearer"
                }

        Adds:
            - expires_at: Calculated expiration timestamp
            - last_update: Current timestamp
            - environment: sim or live
        """
        try:
            # Calculate expiration
            expires_in = tokens.get("expires_in", 1200)  # Default 20min
            expires_at = datetime.now() + timedelta(seconds=expires_in)

            # Enhance token data
            enhanced_tokens = {
                **tokens,
                "expires_at": expires_at.isoformat(),
                "last_update": datetime.now().isoformat(),
                "environment": self.oauth_client.environment
            }

            # Save to user storage
            self.user_fs.write_json(self.tokens_path, enhanced_tokens)
            logger.info(f"✅ Tokens saved for user '{self.user_id}' (expires: {expires_at.isoformat()})")

        except Exception as e:
            logger.error(f"Error saving tokens: {e}")
            raise

    def _load_tokens(self) -> Optional[Dict[str, Any]]:
        """
        Load tokens from user storage.

        Returns:
            Token dict or None if not found
        """
        try:
            tokens = self.user_fs.read_json(self.tokens_path)
            return tokens
        except FileNotFoundError:
            logger.debug(f"No tokens file found for user '{self.user_id}'")
            return None
        except Exception as e:
            logger.error(f"Error loading tokens: {e}")
            return None

    def get_account_key(self) -> Optional[str]:
        """
        Get user's Saxo account key from saved tokens.

        Returns:
            Account key string or None

        Note: Account key is obtained after first successful authentication
        """
        tokens = self._load_tokens()
        if not tokens:
            return None

        # Account key typically returned in user info endpoint
        # For now, assume it's stored in tokens
        return tokens.get("account_key")

    async def disconnect(self) -> None:
        """
        Clear all tokens and disconnect user.

        Use case: Manual logout, security incident

        Note: Always succeeds even if tokens already deleted or expired.
              The goal is to clean up local state, not notify Saxo.
        """
        try:
            # Delete tokens file
            self.user_fs.delete_file(self.tokens_path)
            logger.info(f"✅ User '{self.user_id}' disconnected")
        except FileNotFoundError:
            logger.debug("No tokens to delete - already disconnected")
        except Exception as e:
            # Log error but don't raise - disconnection is always successful
            logger.warning(f"Error during disconnect (ignored): {e}")
            logger.info(f"✅ User '{self.user_id}' marked as disconnected despite error")

    async def cache_positions(self, positions: List[Dict[str, Any]]) -> None:
        """
        Cache positions data for offline fallback.

        Args:
            positions: List of normalized position dicts

        Storage:
            data/users/{user_id}/saxobank/api_cache/positions_YYYYMMDD_HHMMSS.json
        """
        try:
            # Ensure cache directory exists
            cache_dir_path = Path(self.user_fs.get_absolute_path(self.cache_dir))
            cache_dir_path.mkdir(parents=True, exist_ok=True)

            # Generate timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cache_filename = f"{self.cache_dir}/positions_{timestamp}.json"

            # Save
            cache_data = {
                "timestamp": datetime.now().isoformat(),
                "positions": positions,
                "count": len(positions)
            }

            self.user_fs.write_json(cache_filename, cache_data)
            logger.info(f"✅ Cached {len(positions)} positions for user '{self.user_id}'")

            # Cleanup old cache files (keep last 5)
            self._cleanup_old_caches()

        except Exception as e:
            logger.error(f"Error caching positions: {e}")
            # Don't raise - caching failure is non-critical

    async def get_cached_positions(
        self,
        max_age_hours: int = 24
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Retrieve cached positions if available and fresh enough.

        Args:
            max_age_hours: Maximum age of cache in hours (default: 24)

        Returns:
            List of positions or None if no valid cache found
        """
        try:
            cache_dir_path = Path(self.user_fs.get_absolute_path(self.cache_dir))

            if not cache_dir_path.exists():
                logger.debug("No cache directory found")
                return None

            # Find most recent cache file
            cache_files = sorted(cache_dir_path.glob("positions_*.json"), reverse=True)

            if not cache_files:
                logger.debug("No cache files found")
                return None

            latest_cache = cache_files[0]

            # Load cache
            with open(latest_cache, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)

            # Check age
            cache_timestamp = datetime.fromisoformat(cache_data["timestamp"])
            age_hours = (datetime.now() - cache_timestamp).total_seconds() / 3600

            if age_hours > max_age_hours:
                logger.warning(f"⚠️ Cache too old ({age_hours:.1f}h > {max_age_hours}h)")
                return None

            positions = cache_data.get("positions", [])
            logger.info(f"✅ Retrieved {len(positions)} positions from cache (age: {age_hours:.1f}h)")
            return positions

        except Exception as e:
            logger.error(f"Error retrieving cached positions: {e}")
            return None

    def _cleanup_old_caches(self, keep_count: int = 5) -> None:
        """
        Remove old cache files, keeping only the most recent.

        Args:
            keep_count: Number of recent caches to keep
        """
        try:
            cache_dir_path = Path(self.user_fs.get_absolute_path(self.cache_dir))
            cache_files = sorted(cache_dir_path.glob("positions_*.json"), reverse=True)

            # Delete old files
            for old_cache in cache_files[keep_count:]:
                old_cache.unlink()
                logger.debug(f"Deleted old cache: {old_cache.name}")

        except Exception as e:
            logger.error(f"Error cleaning up cache: {e}")

    def get_connection_status(self) -> Dict[str, Any]:
        """
        Get detailed connection status for UI display.

        Returns:
            {
                "status": "connected" | "expired" | "disconnected",
                "connected": bool,  # Deprecated, use "status" instead
                "environment": "sim" | "live",
                "expires_at": str (ISO format),
                "last_update": str (ISO format),
                "account_key": str | None
            }
        """
        tokens = self._load_tokens()

        # No tokens at all
        if not tokens:
            return {
                "status": "disconnected",
                "connected": False,
                "environment": None,
                "expires_at": None,
                "last_update": None,
                "account_key": None
            }

        # Tokens exist but expired
        if not self.is_connected():
            return {
                "status": "expired",
                "connected": False,
                "environment": tokens.get("environment", "sim"),
                "expires_at": tokens.get("expires_at"),
                "last_update": tokens.get("last_update"),
                "account_key": None
            }

        # Valid connection
        return {
            "status": "connected",
            "connected": True,
            "environment": tokens.get("environment", "sim"),
            "expires_at": tokens.get("expires_at"),
            "last_update": tokens.get("last_update"),
            "account_key": tokens.get("account_key")
        }
