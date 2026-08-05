"""Store raw provider responses in PostgreSQL."""

from __future__ import annotations

from alembic import op

revision = "0002_raw_response_jsonb"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema = current_schema()
                  AND table_name = 'generations'
            ) THEN
                ALTER TABLE generations
                    ADD COLUMN IF NOT EXISTS raw_response JSONB;
                ALTER TABLE generations
                    DROP COLUMN IF EXISTS raw_uri;
            END IF;
        END
        $$;
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema = current_schema()
                  AND table_name = 'generations'
            ) THEN
                ALTER TABLE generations
                    ADD COLUMN IF NOT EXISTS raw_uri TEXT;
                ALTER TABLE generations
                    DROP COLUMN IF EXISTS raw_response;
            END IF;
        END
        $$;
        """
    )
