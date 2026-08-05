# Documentation

System design for **evalanche** (`evalctl`). Read these before changing core harness behavior.

| Doc | Audience | Contents |
|-----|----------|----------|
| [Architecture](architecture.md) | All engineers | Components, boundaries, versioning |
| [Data plane](dataplane.md) | Backend / harness | Case → generate → score → report flow |
| [Database schema](schema.md) | Backend / data | Tables, immutability, resume keys |
| [Operations](operations.md) | On-call / local | Compose, CLI, failure modes, PoC |
| [Principles](principles.md) | All engineers | Non-negotiables that shape every change |

**Committed proof of concept:** [`fixtures/poc/`](../fixtures/poc/) — mock-provider E2E reports generated without Ollama/GPU.

Private working notes (if any) live under `docs/private/` and are gitignored.
