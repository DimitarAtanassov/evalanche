# Provider authoring

Providers implement `evalharness.core.protocols.Provider`: resolve an immutable
model revision, declare actual capabilities, generate, batch-embed, and classify
errors. Register one `evalharness.providers` entry point.

The managed runtime wraps adapters with per-provider/model request and token
buckets, a concurrency semaphore, and CLOSED/OPEN/HALF_OPEN circuit breaker.
Adapters must not implement retries; the executor owns retry policy and honors
`Retry-After`. Do not log credentials, authorization headers, prompts, or raw
responses.

OpenAI-compatible endpoints require an explicit `model_revision`; an API model
alias is not a reproducible version. Contract tests should use recorded `respx`
responses and dummy credentials.
