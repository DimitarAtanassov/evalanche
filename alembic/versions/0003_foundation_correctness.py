"""Add correctness-foundation constraints and indexes."""

from __future__ import annotations

from alembic import op

revision = "0003_foundation_correctness"
down_revision = "0002_raw_response_jsonb"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE metric_aggregates ADD COLUMN IF NOT EXISTS metric_config_sha256 TEXT")
    op.execute(
        "UPDATE metric_aggregates SET metric_config_sha256 = 'legacy-unversioned' "
        "WHERE metric_config_sha256 IS NULL"
    )
    op.execute("ALTER TABLE metric_aggregates ALTER COLUMN metric_config_sha256 SET NOT NULL")
    op.execute(
        """
        DELETE FROM metric_aggregates newer USING metric_aggregates older
        WHERE newer.id > older.id
          AND newer.run_id IS NOT DISTINCT FROM older.run_id
          AND newer.metric_name IS NOT DISTINCT FROM older.metric_name
          AND newer.metric_version IS NOT DISTINCT FROM older.metric_version
          AND newer.slice_key = older.slice_key
          AND newer.metric_config_sha256 = older.metric_config_sha256
        """
    )
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conname = 'uq_metric_aggregates_identity'
            ) THEN
                ALTER TABLE metric_aggregates
                    ADD CONSTRAINT uq_metric_aggregates_identity
                    UNIQUE (
                        run_id, metric_name, metric_version, slice_key,
                        metric_config_sha256
                    );
            END IF;
        END
        $$;
        """
    )
    for statement in (
        "CREATE INDEX IF NOT EXISTS ix_generations_run_id ON generations (run_id)",
        "CREATE INDEX IF NOT EXISTS ix_scores_generation_id ON scores (generation_id)",
        "CREATE INDEX IF NOT EXISTS ix_runs_status_started_at ON runs (status, started_at)",
        "CREATE INDEX IF NOT EXISTS ix_generations_run_case_repeat "
        "ON generations (run_id, case_id, repeat_idx)",
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_model_versions_identity "
        "ON model_versions "
        "(provider, model, resolved_version, COALESCE(quantization, ''))",
    ):
        op.execute(statement)


def downgrade() -> None:
    for statement in (
        "DROP INDEX IF EXISTS uq_model_versions_identity",
        "DROP INDEX IF EXISTS ix_scores_generation_id",
        "DROP INDEX IF EXISTS ix_generations_run_id",
        "DROP INDEX IF EXISTS ix_runs_status_started_at",
        "DROP INDEX IF EXISTS ix_generations_run_case_repeat",
        "ALTER TABLE metric_aggregates DROP CONSTRAINT IF EXISTS uq_metric_aggregates_identity",
        "ALTER TABLE metric_aggregates DROP COLUMN IF EXISTS metric_config_sha256",
    ):
        op.execute(statement)
