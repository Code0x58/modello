# Modelling architecture and module boundaries

Use this reference when abstractions, mathematical implementation, dependency
direction, or core/optional/example boundaries change.

## Reuse the modelling relationships

- Check whether new imperative or special-case logic reexpresses relationships
  already represented by Modello components, equations, or native SymPy
  operations. Prefer one authoritative model and reuse the supported core APIs.
- Where the same relationship serves forward, inverse, fitting, reporting, or
  reconciliation uses, vary the observations and components instead of keeping
  parallel mathematical implementations that can disagree.
- Establish behavioral equivalence before replacing a specialized operation
  with core construction. Preserve distinct validation, domain, result, and
  failure contracts; investigate an unfamiliar result type before removing it.

## Keep the layers coherent

- Keep general modelling capabilities in the core, domain extensions in their
  appropriate optional layer, and exploratory/application behavior in examples.
  Check module imports, public exports, dependency groups, documentation
  navigation, and test locations against that same boundary.
- Keep data/GUI libraries optional where intended. A generic core operation
  should not depend on a particular external container when a small adapter or
  documented recipe expresses the integration.
- Follow the existing dependency direction described in the contributor
  documentation. If a local import merely hides a circular dependency, look
  for a shared record or clearer ownership in a lower layer. Preserve local
  imports that have a real optional-loading purpose; assess the dependency,
  not the indentation alone.

Flag a concrete loss of reuse, coupling, or boundary violation. Keep mechanical
import grouping and formatting with the configured tools.
