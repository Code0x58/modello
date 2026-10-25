# Sphinx documentation review

Review documentation as an experienced Python user deciding whether a feature
solves their problem and learning the shortest clear way to use it. Read the
surrounding guide and navigation as well as changed lines. Public API or
workflow changes can need documentation even when no documentation file changed.

## Motivation and progression

- Lead with the class of problem the feature solves, then introduce the concrete
  scenario. Explain why the feature is useful before presenting its mechanics.
- Disclose concepts progressively: introduce names, objects, and prerequisites
  before using them. Lead with the usual convenience API where available and
  a small successful example; move manual alternatives, advanced composition,
  and limitations later or into linked reference pages.
- Assume Python fluency. Explain Modello's behavior and modelling ideas rather
  than teaching Python syntax or exposing internal implementation vocabulary.
- Check that links and page order support the learning progression. A reader
  should not need an unexplained concept from a later section to run an example.
- When related APIs overlap, give short task-based advice on which to choose
  and their meaningful differences, then link to the detailed guides.

## Examples and vocabulary

- Keep examples short, executable, and focused on one useful outcome. Include
  the imports and observations needed to reproduce the displayed result.
- Use model/API properties or symbolic extents where an incidental size or
  index literal would break when inputs change. Keep illustrative constants
  that make the result easy to check; avoid aliases that only repeat an already
  clear property.
- Show the useful contents and meaning of returned objects, including the
  strength of any verdict. Where helpful, show the governing formula or a
  compact input/result table with values and precision that expose the point.
- Look through existing guides and `examples/` before accepting a new model.
  Prefer extending or reusing an existing model when it expresses the same idea.
  A new model should demonstrate a distinct concept that existing examples
  cannot explain as clearly.
- Reuse established public names, model names, and terminology consistently
  across prose, code, and API references. Introduce a new term only for a
  genuinely distinct concept, and define it when first used.

## Interoperability and correctness

- For substantial features, follow the relevant input/output path through
  libraries users already use, especially SymPy. Where applicable, show how to
  supply native objects, obtain ordinary results, and continue with that
  library's operations. Explain meaningful conversions, assumptions, shapes,
  ordering, or unresolved symbolic results at the point they matter.
- Cover the relevant integrations rather than adding an unrelated catalogue of
  libraries. Keep optional integrations optional and consistent with dependency
  groups and the actual supported API.
- Check examples against implementation and tests, including relevant failure
  behavior. Show how users investigate unresolved, under/over-specified, or
  conflicting inputs where relevant, with links to diagnostic guidance. Avoid
  implying a broader guarantee than the code establishes.
- Use the repository's existing executable-example conventions. Run affected
  examples and the warning-free Sphinx HTML build required by `AGENTS.md`.
  Check the rendered result when structure, navigation, or layout changes.

For contributor pages, address maintainers: give concise suite/workflow
invocations, keeping detailed test catalogues beside the tests. Preserve
genuinely distinct required checks instead of listing repetitive per-test runs.

Report concrete reader obstacles and the smallest useful correction. A passing
Sphinx build establishes syntax and references, not a clear learning sequence.
