"""Initial schema."""

from __future__ import annotations

from alembic import op

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    # Tables are created via SQLAlchemy metadata in init_db for Phase 1.
    # Alembic revision exists for migration workflow continuity.


def downgrade() -> None:
    pass
