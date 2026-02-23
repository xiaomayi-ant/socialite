# -*- coding: utf-8 -*-
"""Moltbook Authentication Management"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from .models import MoltbookCredentials, MoltbookRegistration
from .config import MoltbookConfig


import logging
logger = logging.getLogger(__name__)

class MoltbookAuth:
    """Manages Moltbook authentication and credentials"""

    def __init__(self, config: MoltbookConfig):
        """Initialize authentication manager

        Args:
            config: Moltbook configuration
        """
        self.config = config
        self.credentials: Optional[MoltbookCredentials] = None
        self._credentials_path = Path(
            os.path.expanduser(config.credentials_path)
        ).resolve()

    def load_credentials(self) -> bool:
        """Load credentials from file

        Returns:
            True if credentials loaded successfully
        """
        try:
            if not self._credentials_path.exists():
                logger.info("No credentials found at %s", self._credentials_path)
                return False

            with open(self._credentials_path, "r") as f:
                data = json.load(f)

            # Convert timestamps
            if "registered_at" in data and isinstance(data["registered_at"], str):
                data["registered_at"] = datetime.fromisoformat(data["registered_at"])
            if "claimed_at" in data and isinstance(data["claimed_at"], str):
                data["claimed_at"] = datetime.fromisoformat(data["claimed_at"])

            self.credentials = MoltbookCredentials(**data)
            logger.info("Loaded credentials for agent: %s (claimed: %s)", self.credentials.agent_name, self.credentials.claimed)
            return True

        except Exception as e:
            logger.info("❌ Failed to load credentials: %s", e)
            return False

    def save_credentials(self, credentials: MoltbookCredentials) -> bool:
        """Save credentials to file

        Args:
            credentials: Credentials to save

        Returns:
            True if saved successfully
        """
        try:
            # Create directory if it doesn't exist
            self._credentials_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert to dict and handle datetime serialization
            data = credentials.model_dump()
            if "registered_at" in data and isinstance(data["registered_at"], datetime):
                data["registered_at"] = data["registered_at"].isoformat()
            if "claimed_at" in data and isinstance(data["claimed_at"], datetime):
                data["claimed_at"] = data["claimed_at"].isoformat()

            with open(self._credentials_path, "w") as f:
                json.dump(data, f, indent=2)

            self.credentials = credentials
            logger.info("✅ Saved credentials to %s", self._credentials_path)
            return True

        except Exception as e:
            logger.info("❌ Failed to save credentials: %s", e)
            return False

    def save_registration(self, registration: MoltbookRegistration) -> bool:
        """Save registration information

        Args:
            registration: Registration data from API

        Returns:
            True if saved successfully
        """
        credentials = MoltbookCredentials(
            api_key=registration.api_key,
            agent_name=registration.agent_name,
            registered_at=datetime.now(),
            claimed=False,
            claimed_at=None,
        )
        return self.save_credentials(credentials)

    def mark_as_claimed(self) -> bool:
        """Mark agent as claimed

        Returns:
            True if updated successfully
        """
        if not self.credentials:
            logger.info("❌ No credentials loaded")
            return False

        self.credentials.claimed = True
        self.credentials.claimed_at = datetime.now()
        return self.save_credentials(self.credentials)

    def get_api_key(self) -> Optional[str]:
        """Get API key

        Returns:
            API key if available
        """
        if self.credentials:
            return self.credentials.api_key

        # Try to load from config
        if self.config.api_key:
            return self.config.api_key

        return None

    def get_agent_name(self) -> Optional[str]:
        """Get agent name

        Returns:
            Agent name if available
        """
        if self.credentials:
            return self.credentials.agent_name

        if self.config.agent_name:
            return self.config.agent_name

        return None

    def is_registered(self) -> bool:
        """Check if agent is registered

        Returns:
            True if registered
        """
        return self.credentials is not None

    def is_claimed(self) -> bool:
        """Check if agent is claimed

        Returns:
            True if claimed
        """
        return self.credentials is not None and self.credentials.claimed

    def get_auth_header(self) -> Dict[str, str]:
        """Get authorization header for API requests

        Returns:
            Authorization header dict
        """
        api_key = self.get_api_key()
        if not api_key:
            raise ValueError("No API key available")

        return {"Authorization": f"Bearer {api_key}"}

    def clear_credentials(self) -> bool:
        """Clear stored credentials

        Returns:
            True if cleared successfully
        """
        try:
            if self._credentials_path.exists():
                self._credentials_path.unlink()
                logger.info("✅ Cleared credentials from %s", self._credentials_path)

            self.credentials = None
            return True

        except Exception as e:
            logger.info("❌ Failed to clear credentials: %s", e)
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get authentication status

        Returns:
            Status dictionary
        """
        if not self.credentials:
            return {
                "registered": False,
                "claimed": False,
                "agent_name": None,
                "registered_at": None,
            }

        return {
            "registered": True,
            "claimed": self.credentials.claimed,
            "agent_name": self.credentials.agent_name,
            "registered_at": self.credentials.registered_at.isoformat()
            if self.credentials.registered_at
            else None,
            "claimed_at": self.credentials.claimed_at.isoformat()
            if self.credentials.claimed_at
            else None,
        }
