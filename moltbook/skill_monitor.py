# -*- coding: utf-8 -*-
"""Moltbook SKILL.md version monitor — detects API documentation changes."""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import requests

logger = logging.getLogger(__name__)

SKILL_MD_URL = "https://www.moltbook.com/skill.md"
SKILL_MANIFEST_PATH = Path.home() / ".config" / "moltbook" / "skill_manifest.json"


class SkillMonitor:
    """Monitor for Moltbook SKILL.md documentation updates."""

    def __init__(self):
        self.manifest_path = SKILL_MANIFEST_PATH
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

    def fetch_latest_skill(self) -> Optional[str]:
        """Fetch latest SKILL.md from Moltbook."""
        try:
            response = requests.get(SKILL_MD_URL, timeout=10)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.warning(f"Failed to fetch SKILL.md: {e}")
            return None

    def get_file_hash(self, content: str) -> str:
        """Compute SHA256 hash of content."""
        return hashlib.sha256(content.encode()).hexdigest()

    def load_manifest(self) -> Dict:
        """Load stored manifest."""
        if self.manifest_path.exists():
            try:
                return json.loads(self.manifest_path.read_text())
            except:
                pass
        return {
            "version": None,
            "hash": None,
            "last_checked": None,
            "last_changed": None,
        }

    def save_manifest(self, manifest: Dict) -> None:
        """Save manifest to disk."""
        manifest["last_checked"] = datetime.now().isoformat()
        self.manifest_path.write_text(json.dumps(manifest, indent=2))

    def check_for_updates(self) -> Dict[str, any]:
        """Check if SKILL.md has been updated.

        Returns:
            {
                "has_changed": bool,
                "current_hash": str,
                "previous_hash": str,
                "message": str,
                "action": str  # "none" | "review" | "alert"
            }
        """
        latest_content = self.fetch_latest_skill()
        if not latest_content:
            return {
                "has_changed": False,
                "message": "Could not fetch latest SKILL.md",
                "action": "none",
            }

        current_hash = self.get_file_hash(latest_content)
        manifest = self.load_manifest()
        previous_hash = manifest.get("hash")

        if previous_hash is None:
            # First check
            manifest.update(
                {
                    "version": 1,
                    "hash": current_hash,
                    "last_changed": datetime.now().isoformat(),
                }
            )
            self.save_manifest(manifest)
            return {
                "has_changed": True,
                "current_hash": current_hash,
                "previous_hash": None,
                "message": "First time checking SKILL.md",
                "action": "none",
            }

        if current_hash == previous_hash:
            self.save_manifest(manifest)
            return {
                "has_changed": False,
                "current_hash": current_hash,
                "previous_hash": previous_hash,
                "message": "SKILL.md unchanged",
                "action": "none",
            }

        # Changed!
        manifest.update(
            {
                "version": manifest.get("version", 1) + 1,
                "hash": current_hash,
                "last_changed": datetime.now().isoformat(),
            }
        )
        self.save_manifest(manifest)

        logger.warning("⚠️ SKILL.md has been updated!")
        logger.warning(f"   Previous hash: {previous_hash[:12]}...")
        logger.warning(f"   Current hash:  {current_hash[:12]}...")
        logger.warning("   Review: https://www.moltbook.com/skill.md")

        return {
            "has_changed": True,
            "current_hash": current_hash,
            "previous_hash": previous_hash,
            "message": "SKILL.md documentation has changed",
            "action": "review",
        }


# Convenience functions
_monitor = None


def get_monitor() -> SkillMonitor:
    """Get singleton SkillMonitor instance."""
    global _monitor
    if _monitor is None:
        _monitor = SkillMonitor()
    return _monitor


def check_skill_updates() -> Dict:
    """Check for SKILL.md updates (idempotent)."""
    return get_monitor().check_for_updates()
