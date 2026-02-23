# -*- coding: utf-8 -*-
"""
Database Migration Script: v0.2.0 → v0.2.1

Optimizes database schema:
- Merges learning_patterns + style_patterns into unified patterns table
- Adds new fields to social_posts (is_high_value, value_score, quality_with_karma)
- Adds new fields to social_comments (is_high_value, quality_score)
- Adds new own_comments table for comment feedback tracking
- Updates social_users to include karma field

Run this script ONCE before using the new code:
    python -m social_memory.migration_v0.2.1 <db_path>
"""

import sqlite3
import json
import sys
from pathlib import Path
from datetime import datetime


def backup_database(db_path: Path) -> Path:
    """Create a backup of the database before migration"""
    backup_path = db_path.parent / f"{db_path.name}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    try:
        import shutil
        shutil.copy2(db_path, backup_path)
        print(f"✅ Database backed up to: {backup_path}")
        return backup_path
    except Exception as e:
        print(f"❌ Failed to backup database: {e}")
        raise


def migrate_database(db_path: Path) -> bool:
    """Execute the migration"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        print("\n[1/8] Adding karma field to social_users...")
        try:
            cursor.execute("ALTER TABLE social_users ADD COLUMN karma INTEGER DEFAULT 0")
            conn.commit()
            print("✅ Added karma field")
        except sqlite3.OperationalError as e:
            if "duplicate column" in str(e).lower():
                print("⚠️  karma field already exists")
            else:
                raise

        print("\n[2/8] Updating social_posts table...")
        try:
            cursor.execute("ALTER TABLE social_posts ADD COLUMN is_high_value BOOLEAN DEFAULT FALSE")
            cursor.execute("ALTER TABLE social_posts ADD COLUMN value_score REAL DEFAULT 0.0")
            cursor.execute("ALTER TABLE social_posts ADD COLUMN quality_with_karma REAL DEFAULT 0.0")
            conn.commit()
            print("✅ Added is_high_value, value_score, quality_with_karma fields")
        except sqlite3.OperationalError as e:
            if "duplicate column" in str(e).lower():
                print("⚠️  Some fields already exist")
            else:
                raise

        # Rename likes/comments/shares to upvotes/downvotes
        print("\n[3/8] Normalizing social_posts column names...")
        try:
            cursor.execute("ALTER TABLE social_posts RENAME COLUMN likes TO upvotes")
            conn.commit()
            print("✅ Renamed likes → upvotes")
        except Exception as e:
            print(f"⚠️  Column rename skipped: {e}")

        print("\n[4/8] Updating social_comments table...")
        try:
            cursor.execute("ALTER TABLE social_comments ADD COLUMN is_high_value BOOLEAN DEFAULT FALSE")
            cursor.execute("ALTER TABLE social_comments ADD COLUMN quality_score REAL DEFAULT 0.0")
            conn.commit()
            print("✅ Added is_high_value, quality_score fields")
        except sqlite3.OperationalError as e:
            if "duplicate column" in str(e).lower():
                print("⚠️  Some fields already exist")
            else:
                raise

        # Rename likes in social_comments too
        try:
            cursor.execute("ALTER TABLE social_comments RENAME COLUMN likes TO upvotes")
            conn.commit()
            print("✅ Renamed likes → upvotes in social_comments")
        except Exception as e:
            print(f"⚠️  Column rename in social_comments skipped: {e}")

        print("\n[5/8] Creating unified patterns table (merges learning_patterns + style_patterns)...")
        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS patterns (
                    pattern_id TEXT PRIMARY KEY,
                    description TEXT NOT NULL,
                    pattern_source TEXT NOT NULL,
                    confidence REAL DEFAULT 0.5,
                    success_rate REAL DEFAULT 0.0,
                    sample_count INTEGER DEFAULT 0,
                    topics TEXT,
                    effectiveness_score REAL DEFAULT 0.0,
                    usage_count INTEGER DEFAULT 0,
                    last_used TIMESTAMP,
                    example_ids TEXT,
                    metadata TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

            # Migrate data from learning_patterns
            print("   Migrating learning_patterns → patterns...")
            try:
                cursor.execute("""
                    INSERT INTO patterns
                    (pattern_id, description, pattern_source, confidence, success_rate,
                     sample_count, topics, effectiveness_score, usage_count, last_used,
                     example_ids, metadata, created_at, updated_at)
                    SELECT
                        pattern_id, pattern_type, 'learning', adaptability_score, success_rate,
                        usage_count, topics, 0.0, usage_count, last_used,
                        '[]', metadata, created_at, created_at
                    FROM learning_patterns
                    WHERE pattern_id NOT IN (SELECT pattern_id FROM patterns)
                """)
                conn.commit()
                print("   ✅ Migrated learning_patterns data")
            except Exception as e:
                print(f"   ⚠️  Learning patterns migration skipped: {e}")

            # Migrate data from style_patterns
            print("   Migrating style_patterns → patterns...")
            try:
                cursor.execute("""
                    INSERT INTO patterns
                    (pattern_id, description, pattern_source, confidence, success_rate,
                     sample_count, topics, effectiveness_score, usage_count, last_used,
                     example_ids, metadata, created_at, updated_at)
                    SELECT
                        pattern_id, description, 'style', 0.7, success_rate,
                        usage_count, topics, success_rate * 0.8, usage_count, NULL,
                        example_post_ids, metadata, created_at, updated_at
                    FROM style_patterns
                    WHERE pattern_id NOT IN (SELECT pattern_id FROM patterns)
                """)
                conn.commit()
                print("   ✅ Migrated style_patterns data")
            except Exception as e:
                print(f"   ⚠️  Style patterns migration skipped: {e}")

            print("✅ Unified patterns table created and populated")
        except Exception as e:
            print(f"❌ Failed to create patterns table: {e}")
            raise

        print("\n[6/8] Creating own_comments table...")
        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS own_comments (
                    comment_id TEXT PRIMARY KEY,
                    post_id TEXT NOT NULL,
                    comment_text TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    upvotes_current INTEGER DEFAULT 0,
                    upvotes_history TEXT DEFAULT '{}',
                    success BOOLEAN DEFAULT FALSE,
                    feedback_score REAL DEFAULT 0.0,
                    metadata TEXT
                )
            """)
            conn.commit()
            print("✅ Created own_comments table")
        except sqlite3.OperationalError as e:
            if "already exists" in str(e).lower():
                print("⚠️  own_comments table already exists")
            else:
                raise

        print("\n[7/8] Creating indexes for performance...")
        try:
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_social_posts_high_value ON social_posts(is_high_value, value_score DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_social_comments_high_value ON social_comments(is_high_value, quality_score DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_patterns_source ON patterns(pattern_source, effectiveness_score DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_own_comments_post ON own_comments(post_id, created_at DESC)")
            conn.commit()
            print("✅ Created performance indexes")
        except Exception as e:
            print(f"⚠️  Index creation skipped: {e}")

        print("\n[8/8] Verifying migration...")
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        print(f"✅ Database now has {len(tables)} tables:")
        for table in sorted(tables):
            cursor.execute(f"PRAGMA table_info({table})")
            columns = [row[1] for row in cursor.fetchall()]
            print(f"   - {table} ({len(columns)} columns)")

        conn.close()
        return True

    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        return False


def main():
    if len(sys.argv) < 2:
        db_path = Path("./data/social_memory.db")
        print(f"No database path specified, using default: {db_path}")
    else:
        db_path = Path(sys.argv[1])

    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"  Database Migration: v0.2.0 → v0.2.1 (Schema Optimization)")
    print(f"  Database: {db_path}")
    print(f"  Time: {datetime.now().isoformat()}")
    print(f"{'='*70}\n")

    # Create backup
    try:
        backup_path = backup_database(db_path)
    except Exception as e:
        print(f"❌ Backup failed, migration aborted: {e}")
        sys.exit(1)

    # Run migration
    if migrate_database(db_path):
        print(f"\n{'='*70}")
        print("✅ Migration completed successfully!")
        print(f"   Backup saved: {backup_path}")
        print("   You can now use the new v0.2.1 code with optimized schemas")
        print(f"{'='*70}\n")
        sys.exit(0)
    else:
        print(f"\n{'='*70}")
        print("❌ Migration failed!")
        print(f"   Your backup is at: {backup_path}")
        print("   Please restore it and investigate the errors above")
        print(f"{'='*70}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
