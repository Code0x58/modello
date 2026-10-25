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
  Prefer one natural class or container that owns the concept and its related
  members, properties, and behavior when that makes the code simpler.
- Recommend consolidation around a shared meaning or invariant. Do not merge
  merely similar syntax when the contracts differ, or introduce an abstraction
  that makes ordinary use harder to understand.

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
- Look for behavioral coverage of changed contracts, boundaries, and important
  compositions. Favor minimal regressions with an independent expected result
  over tests that mirror implementation or only exercise isolated methods.
- Preserve Python 3.10+ library support and optional dependency boundaries.
  Follow `AGENTS.md` and CI for relevant validation; distinguish observed passes
  from recommended checks and report gaps explicitly.

Return reproducible correctness findings first, followed by concrete contract
or readability issues. Explain the affected scenario and expected behavior.
