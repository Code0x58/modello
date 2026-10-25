# Python, docstring, and comment review

Review the final implementation together with its callers, collaborators,
tests, and public documentation. Prioritize correctness and usable contracts;
base readability suggestions on a specific difficulty for a competent reader.

## Code that reads naturally

- Read the main paths in execution order. Names and structure should make the
  operation clear, like a precise English explanation, without mentally
  decoding abbreviations, distant state, or unnecessary indirection.
- Keep related responsibilities together. Look for duplicated representations
  or operations on the same concept, including duplication across nearby APIs.
  Keep inputs from the same source together so parallel names or records
  cannot silently drift apart.
  Prefer one natural class or container that owns the concept and its related
  members, properties, and behavior when that makes the code simpler.
- Recommend consolidation around a shared meaning or invariant. Do not merge
  merely similar syntax when the contracts differ, or introduce an abstraction
  that makes ordinary use harder to understand.

## Public interfaces

- Keep public modelling interfaces typed and idiomatic with the existing API.
  Prefer model/component instances for declared schemas and relationships;
  mappings can still be appropriate for named collections and data adapters.
- Use named or keyword-only options when positional arguments hide their roles.
  Check that signatures, annotations, and actual accepted values agree.
- For metadata intended to be unique, check that duplicate annotations have an
  explicit policy that catches accidental ambiguity. Preserve the separate
  semantics of repeated equations that intentionally add constraints.

## Docstrings and comments

- Check docstrings against the complete current contract: purpose, meaningful
  parameters and results, mutation or identity behavior, assumptions, errors,
  and limitations where they matter to callers. Describe behavior rather than
  implementation history. Use the established public vocabulary and Sphinx
  cross-references consistently.
- Introduce concepts before using them. Keep examples small and motivated;
  reuse existing example models and show relevant library interoperability.
- Keep comments light. Explain non-obvious reasons, invariants, mathematical
  constraints, or library behavior that a competent reader cannot infer from
  the code. Prefer clearer structure and names to comments translating code
  line by line. Flag stale or misleading explanations.
- Inspect docstrings and comments affected indirectly by a changed method or
  shared record, including autodoc output. Source explanations should agree
  with user guides and actual behavior.

## Correctness and coverage

- Trace interacting methods and classes through the real call path. Check that
  construction, transformations, inspection, and consumption agree on domains,
  shape, ownership, mutation, identity, and failure behavior where relevant.
- At library boundaries, verify the supported input and output types and the
  assumptions made about upstream behavior. For symbolic operations, consider
  unresolved values, exceptional parameter values, and assumptions rather than
  relying only on a generic happy path.
- Use the separately routed test, architecture, and diagnostics references for
  coverage, dependency boundaries, and result/error semantics. Keep supported
  Python versions and validation aligned with `AGENTS.md` and CI; distinguish
  observed passes from recommended checks and report gaps explicitly.

Return reproducible correctness findings first, followed by concrete contract
or readability issues. Explain the affected scenario and expected behavior.
