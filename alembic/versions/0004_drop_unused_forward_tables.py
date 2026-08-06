"""Drop unused judgments, annotations, and embeddings tables.

These tables were reserved for file-primary judge/RAG and durable embedding
storage that never gained a repository write path. Judge and RAG remain
file artifacts; EmbeddingService stays in-memory until a product trigger
requires durable vectors.
"""

from __future__ import annotations

from alembic import op

revision = "0004_drop_unused_forward_tables"
down_revision = "0003_foundation_correctness"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("DROP TABLE IF EXISTS judgments")
    op.execute("DROP TABLE IF EXISTS annotations")
    op.execute("DROP TABLE IF EXISTS embeddings")


def downgrade() -> None:
    # Recreate the prior reserved shapes so a downgrade is not a one-way door.
    # No application path writes these rows; this is schema symmetry only.
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS judgments (
            id BIGSERIAL PRIMARY KEY,
            generation_id BIGINT REFERENCES generations(id),
            compared_generation_id BIGINT REFERENCES generations(id),
            judge_model_version_id BIGINT REFERENCES model_versions(id),
            rubric_name TEXT,
            rubric_version TEXT,
            score INTEGER,
            preference TEXT,
            swap_position INTEGER,
            reasoning TEXT,
            evidence JSONB,
            cost_usd NUMERIC(12, 6)
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS annotations (
            id BIGSERIAL PRIMARY KEY,
            case_id BIGINT REFERENCES cases(id),
            generation_id BIGINT REFERENCES generations(id),
            annotator_id TEXT NOT NULL,
            label JSONB NOT NULL,
            adjudicated BOOLEAN NOT NULL DEFAULT false,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS embeddings (
            id BIGSERIAL PRIMARY KEY,
            content_sha256 TEXT NOT NULL,
            embedding_model_version_id BIGINT NOT NULL REFERENCES model_versions(id),
            vec vector(1024) NOT NULL,
            UNIQUE (content_sha256, embedding_model_version_id)
        )
        """
    )
