# GUI, gallery, and mathematical presentation

Use this reference for visible exploration behavior, model rendering, gallery
entries, or solution plots. Apply the Sphinx reference to shared tutorial prose
and example selection.

## Mathematical fidelity

- Check that displayed formulas reflect the actual symbolic model, with
  recognizable operators, vector notation, grouping, names, and nesting. Raw
  debug strings should not substitute for the intended mathematical view.
- Check displayed constraints, bindings, and field relationships against the
  public model/inspection data. Use the diagnostics reference for provenance and
  nested-path semantics instead of inventing new meanings for the display.

## Concepts and exploration

- Keep a gallery small and order it by the concepts it teaches. Labels should
  identify what a reader learns; examples can then supply the concrete scenario. Use
  consistent practical questions across related entries so users can compare
  the capabilities without relearning the interaction.
- Plot well-named relationships from meaningful models. A demo intended to
  teach curves, areas, surfaces, or solution families should visibly exercise
  that idea; choose a familiar suitable model without adding complexity to an
  example whose purpose is simpler.
- Inspect the rendered states affected by a visible change. Include focused
  screenshots or other visual evidence in the PR when that helps a reviewer
  assess the resulting behavior.
