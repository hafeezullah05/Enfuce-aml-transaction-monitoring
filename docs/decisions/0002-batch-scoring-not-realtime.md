# ADR-0002: Score transactions in daily batches, not in real time

## Status
Accepted

## Context
The data arrives as monthly files of completed transactions with a binary
`Is_laundering` label. Transaction *monitoring* (post-event suspicious-activity
detection and SAR filing) is a different problem from transaction *authorisation*
(sub-second accept/decline at the point of sale).

## Decision
Operate as a scheduled batch job: each day's transactions are scored together,
the top slice by score becomes alerts, and alerts go to an investigator queue.

## Rejected: real-time scoring
- The label and the workflow are post-event; there is no decision to make at
  transaction time.
- Real-time adds a low-latency feature store, online/offline skew risk, and
  streaming infrastructure for no analytical benefit here.
- Entity-history features (trailing-window counts) are naturally a batch
  computation.

## Consequences
- Simple, cheap, reproducible architecture (Part 3).
- Latency to alert is up to ~24h — acceptable for AML monitoring.
- If a future use case needs faster interdiction, revisit with a streaming path
  for a subset of high-risk corridors.
