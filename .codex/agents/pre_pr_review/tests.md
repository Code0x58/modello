# Test design and coverage review

Use this reference for changed tests and for implementation changes whose
contracts need coverage. Review what the tests establish, not their quantity.

## Focus and organization

- Give each case a distinct behavioral purpose. Consolidate overlapping cases
  with meaningful parameterization and reusable fixtures when that makes the
  coverage clearer. Preserve regressions for genuinely different contracts.
- Keep fixtures and helpers proportionate to the behavior they set up. A test
  should reveal its relevant inputs and expected outcome without navigating an
  unnecessary testing framework or several one-use files.
- Make test names and locations reflect the source or example they exercise,
  following the layout in `docs/contributing.rst`. Move tests with reorganized
  modules; composition/scenario tests can naturally span several modules.
- Prefer small exact answers, independent calculations, or equivalent public
  formulations to an oracle that repeats the implementation. Test semantic
  behavior rather than incidental names or presentation details unless those
  details are the contract being reviewed.

## Generality and library boundaries

- Look beyond selected demos for assumptions that accidentally restrict the
  changed capability. Choose representative downstream inputs, boundary cases,
  and feature compositions proportionate to the change, including relevant
  forward/inverse or construction/inspection paths.
- Keep library scenarios independent of gallery menus and curated examples so
  changing the showcase cannot quietly select away unsupported combinations.
  Use the existing scenario suite for cross-feature coverage instead of
  duplicating those cases in every module's tests.
- Verify a claimed upstream defect with a minimal upstream-only reproducer.
  Follow `tests/scenario/README.rst` for diagnostic records and strict xfails.
  Modello defects and documented capability gaps are distinct from upstream
  failures; do not hide a library regression behind an upstream xfail.

Use `AGENTS.md` and CI for required checks. A finite scenario suite supports
the reviewed contracts; passing demos alone does not establish generality.
