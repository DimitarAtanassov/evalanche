# Reports

`evalanche` emits one versioned JSON artifact, a self-contained HTML report,
and JUnit XML for each run. Reporting is read-only; `evalctl runs rescore`
computes idempotent aggregates before report generation.

The HTML report coordinates leadership, research, and engineering views.
Leadership exports omit prompts and raw provider responses. Research includes
confidence methods and metric provenance. Engineering includes latency,
retries, cache behavior, finish reasons, failures, and trace samples. Each
visualization has a neighboring accessible data table.

Coverage uses planned `(case, repeat)` cardinality as its denominator. A partial
run cannot become publishable simply because missing rows are absent.
