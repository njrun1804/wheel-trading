"""
Secure token storage with encryption and validation.
"""

import base64
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from ..utils.logging import get_logger

logger = get_logger(__name__)


class SecureTokenStorage:
    """Secure storage for OAuth tokens with encryption."""

    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize secure token storage.

        Args:
            storage_path: Path to store encrypted tokens. Defaults to ~/.wheel_trading/auth
        """
        self.storage_path = storage_path or Path.home() / ".wheel_trading" / "auth"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.token_file = self.storage_path / "tokens.enc"
        self.key_file = self.storage_path / "key.enc"
        self._cipher = self._get_or_create_cipher()

    def _get_or_create_cipher(self) -> Fernet:
        """Get existing cipher or create new one."""
        if self.key_file.exists():
            with open(self.key_file, "rb") as f:
                key = f.read()
        else:
            # Generate key from machine-specific data
            salt = os.urandom(16)
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            # Use machine ID and user info as password
            password = f"{os.getuid()}-{os.uname().nodename}".encode()
            key = base64.urlsafe_b64encode(kdf.derive(password))

            # Store key with restricted permissions
            self.key_file.write_bytes(key)
            os.chmod(self.key_file, 0o600)

        return Fernet(key)

    def save_tokens(
        self, access_token: str, refresh_token: str, expires_in: int, scope: str = "", **extra_data
    ) -> None:
        """Save tokens securely with expiration tracking.

        Args:
            access_token: OAuth access token
            refresh_token: OAuth refresh token
            expires_in: Token validity in seconds
            scope: OAuth scope
            **extra_data: Additional data to store
        """
        token_data = {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expires_at": (datetime.utcnow() + timedelta(seconds=expires_in)).isoformat(),
            "scope": scope,
            "created_at": datetime.utcnow().isoformat(),
            **extra_data,
        }

        encrypted = self._cipher.encrypt(json.dumps(token_data).encode())
        self.token_file.write_bytes(encrypted)
        os.chmod(self.token_file, 0o600)

        logger.info(
            "save_tokens",
            expires_at=token_data["expires_at"],
            scope=scope,
            has_refresh_token=bool(refresh_token),
        )

    def load_tokens(self) -> Optional[Dict[str, str]]:
        """Load and decrypt tokens if they exist.

        Returns:
            Token data dict or None if not found/invalid
        """
        if not self.token_file.exists():
            logger.warning("load_tokens: Token file not found")
            return None

        try:
            encrypted = self.token_file.read_bytes()
            decrypted = self._cipher.decrypt(encrypted)
            token_data = json.loads(decrypted.decode())

            # Validate token data
            required_fields = ["access_token", "refresh_token", "expires_at"]
            if not all(field in token_data for field in required_fields):
                logger.error("load_tokens: Missing required fields")
                return None

            logger.info(
                f"load_tokens: expires_at={token_data['expires_at']}, "
                f"has_refresh_token={bool(token_data.get('refresh_token'))}"
            )
            return token_data

        except Exception as e:
            logger.error(f"load_tokens: {str(e)}")
            return None

    def is_token_expired(self, buffer_minutes: int = 5) -> bool:
        """Check if stored token is expired or about to expire.

        Args:
            buffer_minutes: Minutes before expiry to consider token expired

        Returns:
            True if expired or about to expire
        """
        token_data = self.load_tokens()
        if not token_data:
            return True

        expires_at = datetime.fromisoformat(token_data["expires_at"])
        buffer = timedelta(minutes=buffer_minutes)
        is_expired = datetime.utcnow() + buffer >= expires_at

        logger.debug(
            "is_token_expired",
            expires_at=expires_at.isoformat(),
            is_expired=is_expired,
            buffer_minutes=buffer_minutes,
        )

        return is_expired

    def clear_tokens(self) -> None:
        """Clear stored tokens (for logout or reset)."""
        if self.token_file.exists():
            self.token_file.unlink()
            logger.info("clear_tokens: Tokens cleared")
