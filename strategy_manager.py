"""
ATLAS Lean v2.0 — Strategy Manager
Versioning of strategy .md files. Handles propose/adopt/reject/rollback.
"""

import logging
import os
import difflib
from datetime import datetime, timezone
from typing import Optional

import config
import db

logger = logging.getLogger(__name__)


class StrategyManager:
    """Manage strategy file versions and lifecycle."""

    def load_active_strategy(self) -> str:
        """Read the current active strategy file."""
        path = os.path.join(config.STRATEGY_DIR, config.ACTIVE_STRATEGY_FILE)
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def save_new_version(
        self,
        name: str,
        text: str,
        created_by: str,
        summary: str,
        changes: str,
    ) -> str:
        """
        Write a new strategy .md file and insert a DB record (PROPOSED).
        Returns the version file name.
        """
        # Write the file
        path = os.path.join(config.STRATEGY_DIR, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

        # Insert DB record
        db.insert_strategy_version(
            version=name,
            created_by=created_by,
            rules_file=name,
            summary=summary,
            changes_json=changes,
        )

        logger.info("New strategy version saved: %s by %s", name, created_by)
        return name

    def adopt_version(self, version_name: str) -> str:
        """
        Adopt a proposed version as the active strategy.
        Archives the currently active version.
        """
        # Verify proposed version exists
        ver = db.get_strategy_version(version_name)
        if ver is None:
            raise ValueError(f"Version {version_name} not found")
        if ver["status"] not in ("PROPOSED", "ARCHIVED"):
            raise ValueError(f"Version {version_name} is {ver['status']}, not PROPOSED")

        path = os.path.join(config.STRATEGY_DIR, version_name)
        if not os.path.exists(path):
            raise ValueError(f"Strategy file not found: {path}")

        # Archive current active
        active = db.get_active_strategy()
        if active:
            db.update_strategy_status(active["version"], "ARCHIVED")

        # Activate new
        db.update_strategy_status(version_name, "ACTIVE")
        config.ACTIVE_STRATEGY_FILE = version_name

        logger.info("Strategy adopted: %s (previous archived)", version_name)
        return f"Adopted {version_name}"

    def reject_version(self, version_name: str) -> str:
        """Reject a proposed strategy version."""
        ver = db.get_strategy_version(version_name)
        if ver is None:
            raise ValueError(f"Version {version_name} not found")

        db.update_strategy_status(version_name, "REJECTED")
        logger.info("Strategy rejected: %s", version_name)
        return f"Rejected {version_name}"

    def rollback(self, version_name: str) -> str:
        """Roll back to a previous strategy version."""
        ver = db.get_strategy_version(version_name)
        if ver is None:
            raise ValueError(f"Version {version_name} not found")

        path = os.path.join(config.STRATEGY_DIR, version_name)
        if not os.path.exists(path):
            raise ValueError(f"Strategy file not found: {path}")

        # Archive current
        active = db.get_active_strategy()
        if active:
            db.update_strategy_status(active["version"], "ARCHIVED")

        db.update_strategy_status(version_name, "ACTIVE")
        config.ACTIVE_STRATEGY_FILE = version_name

        logger.info("Strategy rolled back to: %s", version_name)
        return f"Rolled back to {version_name}"

    def diff_versions(self, name_a: str, name_b: str) -> str:
        """Generate unified diff between two strategy files."""
        path_a = os.path.join(config.STRATEGY_DIR, name_a)
        path_b = os.path.join(config.STRATEGY_DIR, name_b)

        with open(path_a, "r", encoding="utf-8") as f:
            text_a = f.readlines()
        with open(path_b, "r", encoding="utf-8") as f:
            text_b = f.readlines()

        diff = difflib.unified_diff(text_a, text_b, fromfile=name_a, tofile=name_b)
        return "".join(diff) or "(no differences)"

    def get_versions(self) -> list:
        """Return all strategy versions from DB."""
        return db.get_strategy_versions()


# ---------------------------------------------------------------------------
# Self-Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    test_db = "test_strategy.db"
    config.DB_PATH = test_db
    db._conn = None

    try:
        db.init_db()
        sm = StrategyManager()

        print("=== STRATEGY MANAGER TESTS ===")

        # Load active strategy
        text = sm.load_active_strategy()
        print(f"[OK] Active strategy loaded: {len(text)} chars")

        # Save new version
        new_text = text + "\n## Test Amendment\nThis is a test change.\n"
        sm.save_new_version(
            name="strategy_v2_test.md",
            text=new_text,
            created_by="test",
            summary="Test amendment",
            changes="Added test section",
        )
        print("[OK] New version saved: strategy_v2_test.md")

        # Verify in DB
        versions = sm.get_versions()
        assert len(versions) == 1
        assert versions[0]["status"] == "PROPOSED"
        print(f"[OK] Version in DB: {versions[0]['version']} [{versions[0]['status']}]")

        # Register v1 as active in DB
        db.insert_strategy_version(
            version="strategy_v1.md",
            created_by="system",
            rules_file="strategy_v1.md",
            summary="Initial strategy",
            changes_json="Initial",
        )
        db.update_strategy_status("strategy_v1.md", "ACTIVE")

        # Adopt v2
        sm.adopt_version("strategy_v2_test.md")
        assert config.ACTIVE_STRATEGY_FILE == "strategy_v2_test.md"
        print(f"[OK] Adopted: active is now {config.ACTIVE_STRATEGY_FILE}")

        # Check v1 is archived
        v1 = db.get_strategy_version("strategy_v1.md")
        assert v1["status"] == "ARCHIVED"
        print("[OK] Previous version archived")

        # Diff
        diff = sm.diff_versions("strategy_v1.md", "strategy_v2_test.md")
        assert "Test Amendment" in diff
        print(f"[OK] Diff generated: {len(diff)} chars")

        # Rollback
        sm.rollback("strategy_v1.md")
        assert config.ACTIVE_STRATEGY_FILE == "strategy_v1.md"
        print(f"[OK] Rolled back to {config.ACTIVE_STRATEGY_FILE}")

        # Cleanup test file
        os.remove("strategy_v2_test.md")

        print("\n=== ALL STRATEGY MANAGER TESTS PASSED ===")

    finally:
        config.ACTIVE_STRATEGY_FILE = "strategy_v1.md"
        if db._conn:
            db._conn.close()
            db._conn = None
        if os.path.exists(test_db):
            os.remove(test_db)
