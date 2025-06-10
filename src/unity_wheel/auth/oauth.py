"""
OAuth2 flow implementation with automatic browser handling.
"""

import asyncio
import secrets
import socket
import webbrowser
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.parse import parse_qs, urlencode

import aiohttp
from aiohttp import web

from src.config.loader import get_config

from ..utils.logging import get_logger
from .exceptions import AuthError, InvalidCredentialsError

logger = get_logger(__name__)


class OAuth2Handler:
    """Handles OAuth2 authorization flow with browser automation."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        redirect_uri: str = "https://127.0.0.1:8182/callback",
        auth_url: str = "https://api.schwabapi.com/v1/oauth/authorize",
        token_url: str = "https://api.schwabapi.com/v1/oauth/token",
    ):
        """Initialize OAuth2 handler.

        Args:
            client_id: OAuth client ID
            client_secret: OAuth client secret
            redirect_uri: Callback URL (must match registered URL)
            auth_url: Authorization endpoint
            token_url: Token exchange endpoint
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.auth_url = auth_url
        self.token_url = token_url
        self._server = None
        self._auth_code_future = None

    def _get_free_port(self) -> int:
        """Find a free port for the callback server."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port

    async def _handle_callback(self, request: web.Request) -> web.Response:
        """Handle OAuth callback with authorization code."""
        code = request.query.get("code")
        error = request.query.get("error")

        if error:
            error_desc = request.query.get("error_description", "Unknown error")
            self._auth_code_future.set_exception(
                InvalidCredentialsError(f"OAuth error: {error} - {error_desc}")
            )
            return web.Response(
                text=f"<html><body><h1>Authentication Failed</h1><p>{error_desc}</p></body></html>",
                content_type="text/html",
            )

        if not code:
            self._auth_code_future.set_exception(AuthError("No authorization code received"))
            return web.Response(
                text="<html><body><h1>Authentication Failed</h1><p>No authorization code received</p></body></html>",
                content_type="text/html",
            )

        self._auth_code_future.set_result(code)

        return web.Response(
            text="""
            <html>
            <body style="font-family: sans-serif; text-align: center; padding: 50px;">
                <h1 style="color: #4CAF50;">Authentication Successful!</h1>
                <p>You can close this window and return to the application.</p>
                <script>setTimeout(() => window.close(), 3000);</script>
            </body>
            </html>
            """,
            content_type="text/html",
        )

    async def _run_callback_server(self, port: int) -> None:
        """Run the callback server to receive auth code."""
        import os
        import ssl
        import tempfile

        app = web.Application()
        app.router.add_get("/callback", self._handle_callback)

        runner = web.AppRunner(app)
        await runner.setup()

        # Create SSL context for HTTPS
        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)

        # Create self-signed certificate for localhost
        try:
            # Try to use existing cert if available
            cert_path = Path.home() / ".wheel_trading" / "localhost.pem"
            key_path = Path.home() / ".wheel_trading" / "localhost.key"

            if not cert_path.exists() or not key_path.exists():
                # Generate self-signed certificate
                os.makedirs(cert_path.parent, exist_ok=True)
                os.system(
                    f'openssl req -x509 -newkey rsa:4096 -nodes -out {cert_path} -keyout {key_path} -days 365 -subj "/CN=127.0.0.1" 2>/dev/null'
                )

            ssl_context.load_cert_chain(str(cert_path), str(key_path))
        except Exception as e:
            logger.warning(f"SSL setup failed, falling back to HTTP: {e}")
            ssl_context = None

        site = web.TCPSite(runner, "127.0.0.1", port, ssl_context=ssl_context)
        await site.start()

        self._server = runner

    async def _stop_callback_server(self) -> None:
        """Stop the callback server."""
        if self._server:
            await self._server.cleanup()
            self._server = None

    async def authorize(self, scope: str = "AccountsRead MarketDataRead") -> str:
        """Start OAuth2 authorization flow.

        Args:
            scope: OAuth scopes to request

        Returns:
            Authorization code

        Raises:
            AuthError: If authorization fails
        """
        # Extract port from redirect URI
        port = int(self.redirect_uri.split(":")[-1].split("/")[0])

        # Start callback server
        self._auth_code_future = asyncio.Future()
        await self._run_callback_server(port)

        try:
            # Generate state for CSRF protection
            state = secrets.token_urlsafe(32)

            # Build authorization URL
            auth_params = {
                "response_type": "code",
                "client_id": self.client_id,
                "redirect_uri": self.redirect_uri,
                "scope": scope,
                "state": state,
            }
            auth_url_full = f"{self.auth_url}?{urlencode(auth_params)}"

            logger.info(
                "authorize", action="opening_browser", url=auth_url_full[:50] + "...", scope=scope
            )

            # Open browser
            if not webbrowser.open(auth_url_full):
                logger.error("authorize", error="Failed to open browser")
                raise AuthError(
                    "Failed to open browser. Please manually navigate to the URL logged above."
                )

            # Wait for callback with timeout
            try:
                config = get_config()
                # Use a longer timeout for OAuth flow (5 minutes)
                oauth_timeout = max(300, config.data.api_timeouts.total * 5)
                code = await asyncio.wait_for(self._auth_code_future, timeout=oauth_timeout)
                logger.info("authorize", status="success", has_code=bool(code))
                return code

            except asyncio.TimeoutError:
                raise AuthError(
                    "Authorization timeout. Please complete the login within 5 minutes."
                )

        finally:
            await self._stop_callback_server()

    async def exchange_code_for_tokens(self, code: str) -> Dict[str, any]:
        """Exchange authorization code for access tokens.

        Args:
            code: Authorization code from OAuth flow

        Returns:
            Token response dict with access_token, refresh_token, etc.

        Raises:
            AuthError: If token exchange fails
        """
        token_data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.redirect_uri,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }

        async with aiohttp.ClientSession() as session:
            try:
                config = get_config()
                timeout = aiohttp.ClientTimeout(total=config.data.api_timeouts.total)
                async with session.post(
                    self.token_url, data=token_data, timeout=timeout
                ) as response:
                    response_data = await response.json()

                    if response.status != 200:
                        error = response_data.get("error", "Unknown error")
                        error_desc = response_data.get("error_description", "")
                        raise InvalidCredentialsError(
                            f"Token exchange failed: {error} - {error_desc}"
                        )

                    # Validate required fields
                    required = ["access_token", "refresh_token", "expires_in"]
                    if not all(field in response_data for field in required):
                        raise AuthError("Invalid token response - missing required fields")

                    logger.info(
                        "exchange_code_for_tokens",
                        status="success",
                        expires_in=response_data.get("expires_in"),
                        scope=response_data.get("scope", ""),
                    )

                    return response_data

            except aiohttp.ClientError as e:
                logger.error("exchange_code_for_tokens", error=str(e))
                raise AuthError(f"Network error during token exchange: {e}")

    async def refresh_access_token(self, refresh_token: str) -> Dict[str, any]:
        """Refresh access token using refresh token.

        Args:
            refresh_token: Valid refresh token

        Returns:
            New token response dict

        Raises:
            AuthError: If refresh fails
        """
        token_data = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }

        async with aiohttp.ClientSession() as session:
            try:
                config = get_config()
                timeout = aiohttp.ClientTimeout(total=config.data.api_timeouts.total)
                async with session.post(
                    self.token_url, data=token_data, timeout=timeout
                ) as response:
                    response_data = await response.json()

                    if response.status == 401:
                        # Refresh token is invalid - need new authorization
                        raise InvalidCredentialsError(
                            "Refresh token expired. Please re-authenticate."
                        )

                    if response.status != 200:
                        error = response_data.get("error", "Unknown error")
                        raise AuthError(f"Token refresh failed: {error}")

                    logger.info(
                        "refresh_access_token",
                        status="success",
                        expires_in=response_data.get("expires_in"),
                    )

                    return response_data

            except aiohttp.ClientError as e:
                logger.error("refresh_access_token", error=str(e))
                raise AuthError(f"Network error during token refresh: {e}")
