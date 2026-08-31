# ADR-0001: Record architecture decisions

## Status
Accepted

## Context
This case study is judged on technical decision-making, not model score. The
decisions matter more than the code, and they need to be reviewable without
walking through every file.

## Decision
Every significant choice gets a short ADR in `docs/decisions/`, numbered and
dated, using the format: Context / Decision / Consequences (and Rejected
alternatives where relevant). Code comments reference the ADR number.

## Consequences
- The reasoning is auditable — which also matches how a regulated AML system
  must document model-governance decisions.
- ADRs are immutable; a reversal is a new ADR that supersedes the old one.
