"""Initial Alembic-owned schema."""

from __future__ import annotations

from alembic import op
from evalharness.store.models import Base

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    # Metadata is executed by the migration, never by application runtime.
    # checkfirst makes this baseline adopt installations created before Alembic
    # became the sole schema owner.
    Base.metadata.create_all(bind=op.get_bind(), checkfirst=True)


def downgrade() -> None:
    Base.metadata.drop_all(bind=op.get_bind(), checkfirst=True)
