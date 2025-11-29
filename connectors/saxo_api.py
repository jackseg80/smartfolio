"""
SaxoBank OpenAPI OAuth2 Client

Implements Authorization Code Flow with PKCE for secure authentication.
Supports both Simulation and Live environments with dynamic configuration.

References:
- Saxo OpenAPI Docs: https://www.developer.saxo/openapi
- OAuth2 RFC 7636 (PKCE): https://tools.ietf.org/html/rfc7636
"""
from __future__ import annotations

import os
import base64
import hashlib
import secrets
import logging
import httpx
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from urllib.parse import urlencode
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()


class SaxoOAuthClient:
    """
    OAuth2 client for SaxoBank OpenAPI with PKCE support.

    Handles:
    - Authorization URL generation
    - Code-to-token exchange
    - Token refresh
    - API data fetching (positions, balances, transactions)

    Environment Selection:
    - Simulation: For development (demo account)
    - Live: For production (real trading account)

    Set via: SAXO_ENVIRONMENT=sim or SAXO_ENVIRONMENT=live
    """

    def __init__(self):
        """Initialize client with environment-specific configuration."""
        self.environment = os.getenv("SAXO_ENVIRONMENT", "sim").lower()

        if self.environment == "live":
            # Production configuration
            self.auth_url = "https://live.logonvalidation.net/authorize"
            self.token_url = "https://live.logonvalidation.net/token"
            self.api_base = "https://gateway.saxobank.com/openapi"
            self.client_id = os.getenv("SAXO_LIVE_CLIENT_ID", "")
            self.client_secret = os.getenv("SAXO_LIVE_CLIENT_SECRET", "")
            logger.info("🔴 SaxoOAuthClient initialized in LIVE mode")
        else:
            # Simulation configuration (default)
            self.auth_url = "https://sim.logonvalidation.net/authorize"
            self.token_url = "https://sim.logonvalidation.net/token"
            self.api_base = "https://gateway.saxobank.com/sim/openapi"
            self.client_id = os.getenv("SAXO_SIM_CLIENT_ID", "")
            self.client_secret = os.getenv("SAXO_SIM_CLIENT_SECRET", "")
            logger.info("🟢 SaxoOAuthClient initialized in SIMULATION mode")

        # Get redirect URIs from env (supports multiple)
        redirect_uris_str = os.getenv("SAXO_REDIRECT_URI", "http://localhost:8080/api/saxo/callback")
        self.redirect_uris = [uri.strip() for uri in redirect_uris_str.split(",")]
        self.redirect_uri = self.redirect_uris[0]  # Use first as default

        if not self.client_id or not self.client_secret:
            logger.warning(f"⚠️ Saxo {self.environment.upper()} credentials not configured in .env")

    def generate_pkce_pair(self) -> Dict[str, str]:
        """
        Generate PKCE code_verifier and code_challenge.

        Returns:
            {"code_verifier": "...", "code_challenge": "..."}

        PKCE (RFC 7636) prevents authorization code interception attacks.
        """
        # Generate random 43-128 character string
        code_verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('utf-8').rstrip('=')

        # SHA256 hash of verifier
        code_challenge = base64.urlsafe_b64encode(
            hashlib.sha256(code_verifier.encode('utf-8')).digest()
        ).decode('utf-8').rstrip('=')

        return {
            "code_verifier": code_verifier,
            "code_challenge": code_challenge
        }

    def get_authorization_url(self, state: str, code_challenge: str) -> str:
        """
        Generate OAuth2 authorization URL for user login.

        Args:
            state: Random string to prevent CSRF attacks
            code_challenge: PKCE challenge from generate_pkce_pair()

        Returns:
            Full authorization URL (user navigates to this)

        Example:
            https://sim.logonvalidation.net/authorize?response_type=code&client_id=...
        """
        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256"
        }

        url = f"{self.auth_url}?{urlencode(params)}"
        logger.debug(f"📝 Authorization URL generated for state '{state[:8]}...'")
        return url

    async def exchange_code_for_tokens(
        self,
        code: str,
        code_verifier: str
    ) -> Dict[str, Any]:
        """
        Exchange authorization code for access + refresh tokens.

        Args:
            code: Authorization code from callback (?code=...)
            code_verifier: PKCE verifier (must match challenge)

        Returns:
            {
                "access_token": "...",
                "refresh_token": "...",
                "expires_in": 1200,  # seconds (20 min)
                "token_type": "Bearer"
            }

        Raises:
            httpx.HTTPStatusError: If token exchange fails
        """
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.redirect_uri,
            "code_verifier": code_verifier,
            "client_id": self.client_id,
            "client_secret": self.client_secret
        }

        async with httpx.AsyncClient() as client:
            logger.info(f"🔄 Exchanging code for tokens (env: {self.environment})")
            response = await client.post(
                self.token_url,
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            response.raise_for_status()

            tokens = response.json()
            logger.info("✅ Token exchange successful")
            return tokens

    async def refresh_access_token(self, refresh_token: str) -> Dict[str, Any]:
        """
        Refresh expired access token using refresh token.

        Args:
            refresh_token: Valid refresh token from previous auth

        Returns:
            {
                "access_token": "...",
                "refresh_token": "...",  # New refresh token
                "expires_in": 1200,
                "token_type": "Bearer"
            }

        Raises:
            httpx.HTTPStatusError: If refresh fails (401 = user must re-auth)
        """
        data = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": self.client_id,
            "client_secret": self.client_secret
        }

        async with httpx.AsyncClient() as client:
            logger.info(f"🔄 Refreshing access token (env: {self.environment})")
            response = await client.post(
                self.token_url,
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )

            if response.status_code == 401:
                logger.warning("⚠️ Refresh token expired - user must reconnect")
                raise httpx.HTTPStatusError(
                    "Refresh token expired",
                    request=response.request,
                    response=response
                )

            response.raise_for_status()
            tokens = response.json()
            logger.info("✅ Token refresh successful")
            return tokens

    async def get_positions(
        self,
        access_token: str,
        account_key: str
    ) -> List[Dict[str, Any]]:
        """
        Fetch current positions from Saxo OpenAPI.

        Endpoint: GET /port/v1/positions

        Args:
            access_token: Valid Bearer token
            account_key: User's account identifier

        Returns:
            List of positions:
            [{
                "NetPositionId": "...",
                "AssetType": "Stock",
                "Amount": 100,
                "MarketValue": 15000.0,
                "DisplayAndFormat": {
                    "Symbol": "AAPL:xnas",
                    "Description": "Apple Inc."
                },
                "Isin": "US0378331005",
                ...
            }]
        """
        url = f"{self.api_base}/port/v1/positions"
        params = {"AccountKey": account_key}
        headers = {"Authorization": f"Bearer {access_token}"}

        async with httpx.AsyncClient(timeout=30.0) as client:
            logger.info(f"📊 Fetching positions for account {account_key[:8]}...")
            response = await client.get(url, params=params, headers=headers)
            response.raise_for_status()

            data = response.json()
            positions = data.get("Data", [])
            logger.info(f"✅ Retrieved {len(positions)} positions")
            return positions

    async def get_balances(
        self,
        access_token: str,
        account_key: str
    ) -> Dict[str, Any]:
        """
        Fetch account balances and cash positions.

        Endpoint: GET /port/v1/balances

        Args:
            access_token: Valid Bearer token
            account_key: User's account identifier

        Returns:
            {
                "TotalValue": 250000.0,
                "CashBalance": 50000.0,
                "Currency": "EUR",
                "CurrencyDecimals": 2,
                ...
            }
        """
        url = f"{self.api_base}/port/v1/balances"
        params = {"AccountKey": account_key}
        headers = {"Authorization": f"Bearer {access_token}"}

        async with httpx.AsyncClient(timeout=30.0) as client:
            logger.info(f"💰 Fetching balances for account {account_key[:8]}...")
            response = await client.get(url, params=params, headers=headers)
            response.raise_for_status()

            balances = response.json()
            logger.info(f"✅ Retrieved balances (Total: {balances.get('TotalValue', 0)})")
            return balances

    async def get_transactions(
        self,
        access_token: str,
        account_key: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch transaction history for P&L calculation.

        Endpoint: GET /hist/v1/transactions

        Args:
            access_token: Valid Bearer token
            account_key: User's account identifier
            from_date: Start date (ISO format: YYYY-MM-DD)
            to_date: End date (ISO format: YYYY-MM-DD)

        Returns:
            List of transactions:
            [{
                "TransactionId": "...",
                "TradeDate": "2025-11-15",
                "AssetType": "Stock",
                "Amount": 100,
                "Price": 150.0,
                "NetAmount": -15000.0,
                "Currency": "USD",
                ...
            }]

        Note: If dates not specified, defaults to last 30 days
        """
        url = f"{self.api_base}/hist/v1/transactions"

        # Default to last 30 days if not specified
        if not to_date:
            to_date = datetime.now().strftime("%Y-%m-%d")
        if not from_date:
            from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        params = {
            "AccountKey": account_key,
            "FromDate": from_date,
            "ToDate": to_date
        }
        headers = {"Authorization": f"Bearer {access_token}"}

        async with httpx.AsyncClient(timeout=30.0) as client:
            logger.info(f"📜 Fetching transactions {from_date} to {to_date}...")
            response = await client.get(url, params=params, headers=headers)
            response.raise_for_status()

            data = response.json()
            transactions = data.get("Data", [])
            logger.info(f"✅ Retrieved {len(transactions)} transactions")
            return transactions


# Helper functions for PKCE state management
# (Redis storage handled by saxo_auth_router.py)

def generate_state() -> str:
    """Generate random state string for CSRF protection."""
    return secrets.token_urlsafe(32)
