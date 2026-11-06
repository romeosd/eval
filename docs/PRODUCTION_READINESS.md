# Production readiness

This repository is an assessment submission built the way production software is built. It
is **not** a deployed service. The decision boundary (what the model may decide, how its
output is verified, how failures resolve, how runs are recorded and reproduced) is built to
a production standard. Everything around that boundary is documented as a path, not built.

| Concern | Built | Designed, not built | Out of scope |
|---|---|---|---|
| Safety of decisions | Fail-closed outcomes; nine deterministic checks; claim-support judge; completeness gate; unit-aware provenance for calculation inputs; injection screening with quarantine | Human review of sampled decisions; policy-owner approval of changes | Regulatory sign-off |
| Determinism | Deterministic orchestration, ordering and data generation; pinned-snapshot enforcement; record/replay; optional consensus gate | Cross-provider fallback with equivalence checks | Bit-identical model sampling (not available from the API) |
| Reliability | Timeouts, SDK retries, budget guard, atomic writes, resumable runs, verifier-outage handling | Queue with idempotent job ids, dead-letter handling, circuit breakers | Autoscaling |
| Observability | Per-case audit trail, cost and latency per call, resolved model ids, run manifests | Metrics export, dashboards, alerting, drift monitors | On-call process |
| Reproducibility | Immutable runs, snapshot of policy index and prompts, git SHA and dirty flag | Signed, retained audit store | Long-term retention policy |
| Quality gates | Strict typing, lint, format, 147 offline tests, CI workflow, label audit | Evaluation gates on pull requests using recorded cassettes | Load testing |
| Data protection | `store=false`, no secrets in config, `.env` ignored | PII detection and redaction, retention limits, access control on audit logs | Data-residency controls |
| Interfaces | CLI | HTTP API, review UI, webhook callbacks | Authentication and multi-tenancy |
| Change management | Versioned prompts with hashes; behaviour fingerprint guards resume | Shadow mode and staged rollout of prompt or model changes | Release trains |

## The honest gaps

- Reliability is unproven: nothing has run against the live API.
- Outcomes are model-decided and then verified; they are not computed by a rules engine.
  Extracting verified rules is the production path for stable, high-volume policies, with a
  policy owner approving the rules. It was not chosen here because policy wording includes
  qualitative terms that do not compile to rules and the sample is too small to validate
  extracted rules.
- The judge and the provenance check are partial controls. Provenance verifies that a number
  and its unit exist in the cited source, not that it is the right clause.
