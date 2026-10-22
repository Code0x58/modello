"use strict";
// Keep chart settings and checked samples together across edits and asynchronous requests.
class ChartSession {
  data = null;
  preferences = null;
  pending = null;
  initialised = false;
  revision = 0;

  accept(data, options) {
    this.revision++;
    const changed =
      options.edit ||
      options.scenario !== undefined ||
      options.bounds ||
      options.point ||
      options.case !== undefined ||
      data.raw;
    if (!options.fromPlot && changed) this.data = null;
    if (options.plot !== undefined) this.usePreset(options.plot);
    else if (!this.initialised) this.usePreset(data.plot);
    this.initialised = true;
    if (data.raw) this.pending = null;
  }
  usePreset(preset) {
    if (preset === undefined) return;
    this.preferences = preset;
    this.pending = preset;
    this.data = null;
  }
  takePending() {
    const preset = this.pending;
    this.pending = null;
    return preset;
  }
  async sample(preferences, context) {
    this.preferences = preferences;
    const revision = this.revision;
    const { controls, windows, output } = preferences;
    try {
      const response = await fetch("/api/plot", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...context, controls, windows, output }),
      });
      const result = await response.json();
      if (revision !== this.revision) return null;
      if (response.ok) this.data = result;
      return { ok: response.ok, result };
    } catch (error) {
      if (revision !== this.revision) return null;
      throw error;
    }
  }
}
// A selected point and a pinned comparison never become model observations.
const familyUI = (() => {
  const chart = new ChartSession();
  let bounds = {},
    selection = null,
    baseline = null;
  const get = (id) => document.getElementById(id);
  const button = (label, action, className = "ghost") => {
    const element = node("button", className, label);
    element.type = "button";
    element.addEventListener("click", action);
    return element;
  };
  const canonical = () =>
    state.fields.filter(
      (field) =>
        field.path ===
        (state.highlights.find((path) => field.aliases.includes(path)) ??
          (field.aliases.includes(state.initial_field)
            ? state.initial_field
            : field.aliases[0])),
    );
  const scalarText = (view, field) =>
    `${view.approximate ? "≈" : ""}${view.display_value || view.value}${field.unit ? " " + field.unit : ""}`;
  function request(options = {}) {
    return {
      bounds: options.bounds ?? bounds,
      case: options.case ?? null,
      point: options.point ?? null,
    };
  }
  function accepted(data, options) {
    bounds = options.bounds ?? bounds;
    selection = data.exploration?.selection ?? null;
    chart.accept(data, options);
  }
  function currentSelection() {
    return selection?.kind === "case"
      ? { case: selection.index }
      : selection?.kind === "point"
        ? { point: selection.coordinates }
        : {};
  }
  function removeBound(field) {
    const next = withoutAliases(bounds, field);
    return refresh(bindings, { bounds: next, edit: true });
  }
  function rangeEditor(field) {
    const container = section("Bound this quantity");
    const form = node("form", "range-form");
    const inputs = ["At least", "At most"].map((title, index) => {
      const label = node("label", "range-label", title);
      const input = node("input");
      input.type = "text";
      input.value = field.bounds[index];
      input.placeholder = "No bound";
      input.setAttribute("aria-label", `${title} ${field.path}`);
      label.append(input);
      form.append(label);
      return input;
    });
    const actions = node("div", "form-actions");
    const submit = node("button", "primary", "Apply bounds");
    submit.type = "submit";
    actions.append(submit);
    if (field.bounds.some(Boolean))
      actions.append(button("Remove bounds", () => removeBound(field)));
    form.append(
      actions,
      node(
        "p",
        "input-help",
        field.editable
          ? "Inclusive bounds retain the equations. Applying a range releases this quantity’s exact input; other observations and inherited filters stay active."
          : "Bounds narrow the possibilities. This recorded observation and every inherited filter remain active.",
      ),
    );
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      const endpoints = inputs.map((input) => input.value.trim() || null);
      const nextBounds = withoutAliases(bounds, field);
      let nextBindings = { ...bindings };
      if (endpoints.some((value) => value !== null)) {
        nextBounds[field.path] = endpoints;
        if (field.kind === "input") {
          nextBindings = withoutAliases(nextBindings, field);
          nextBindings[field.path] = null;
        }
      }
      refresh(nextBindings, { bounds: nextBounds, edit: true });
    });
    container.append(form);
    return container;
  }
  function familyValues(field) {
    if (!field.possible_values) return null;
    const container = section("Possible across the whole family");
    container.append(node("p", "set-summary", field.possible_values.text));
    const exact = node("details", "exact-value");
    exact.append(
      node("summary", "", "Exact set"),
      node("p", "set-exact", field.possible_values.exact),
    );
    container.append(exact);
    return container;
  }
  function caseTable(info) {
    const container = get("family-cases");
    container.replaceChildren();
    if (!info.cases?.length) return;
    const paths = [
      ...new Set([
        ...info.coordinates.map((coordinate) => coordinate.path),
        ...state.highlights,
        ...info.distinguishing,
      ]),
    ]
      .filter((path) => info.cases[0].values[path])
      .slice(0, 5);
    if (!paths.length)
      paths.push(...Object.keys(info.cases[0].values).slice(0, 3));
    const table = node("table", "family-table");
    const head = node("thead"),
      header = node("tr");
    header.append(node("th", "", "Case"));
    paths.forEach((path) =>
      header.append(node("th", "", fieldsByPath.get(path).title)),
    );
    head.append(header);
    const body = node("tbody");
    info.cases.forEach((entry) => {
      const row = node(
        "tr",
        entry.index === selection?.index ? "selected-case" : "",
      );
      const label = node("td");
      const inspect = button(`Case ${entry.index + 1}`, () =>
        refresh(bindings, { case: entry.index }),
      );
      inspect.setAttribute(
        "aria-pressed",
        String(entry.index === selection?.index && selection?.kind === "case"),
      );
      label.append(inspect);
      row.append(label);
      paths.forEach((path) => {
        const cell = node(
          "td",
          "",
          scalarText(entry.values[path], fieldsByPath.get(path)),
        );
        cell.title = entry.values[path].value;
        row.append(cell);
      });
      body.append(row);
    });
    table.append(head, body);
    container.append(table);
    if (info.cases.length > 1 && selection)
      container.append(
        button("Show the whole family", () => refresh(bindings)),
      );
  }
  function compare() {
    const container = get("family-comparison");
    container.replaceChildren();
    if (state.status === "checked")
      container.append(
        button("Pin this checked case", () => {
          baseline = {
            fields: canonical().map((field) => ({ ...field })),
            observations: canonical()
              .filter((field) => field.kind === "input")
              .map((field) => `${field.title}: ${field.input}`),
            conditions: state.exploration?.conditions ?? [],
          };
          compare();
        }),
      );
    if (!baseline) return;
    container.append(
      button("Clear comparison", () => {
        baseline = null;
        compare();
      }),
    );
    const context = node("details", "comparison-context");
    context.append(
      node("summary", "", "Pinned observations and filters"),
      node(
        "p",
        "set-exact",
        [...baseline.observations, ...baseline.conditions].join(" · "),
      ),
    );
    container.append(context);
    if (state.status !== "checked") {
      container.append(
        node(
          "p",
          "input-help",
          "Choose a checked case to compare with the pinned case.",
        ),
      );
      return;
    }
    const table = node("table", "family-table");
    const head = node("thead"),
      header = node("tr");
    ["Quantity", "Pinned", "Current"].forEach((label) =>
      header.append(node("th", "", label)),
    );
    head.append(header);
    const body = node("tbody");
    baseline.fields.forEach((before) => {
      const after = fieldsByPath.get(before.path);
      if (
        !after ||
        (before.value === after.value &&
          !state.highlights.includes(before.path))
      )
        return;
      const row = node("tr");
      const name = node("td");
      name.append(button(after.title, () => choose(after.path)));
      row.append(
        name,
        node("td", "", scalarText(before, before)),
        node("td", "", scalarText(after, after)),
      );
      body.append(row);
    });
    if (!body.children.length) {
      container.append(
        node(
          "p",
          "input-help",
          "The visible quantities match the pinned case.",
        ),
      );
      return;
    }
    table.append(head, body);
    container.append(table);
  }
  function coordinateForms(info) {
    const container = get("family-controls");
    container.replaceChildren();
    const coordinates = info.coordinates ?? [];
    if (!coordinates.length || ["empty", "point", "cases"].includes(info.kind))
      return;
    if (
      coordinates.length !== info.coordinate_count ||
      coordinates.some((coordinate) => !coordinate.previewable)
    ) {
      container.append(
        node(
          "p",
          "input-help",
          "Some remaining coordinates are background quantities. Their restrictions remain active.",
        ),
      );
      return;
    }
    container.append(pointPreview(coordinates));
    const plotCoordinates = coordinates.filter((coordinate) => coordinate.real);
    if (!plotCoordinates.length) {
      container.append(
        node(
          "p",
          "input-help",
          "Charts need real coordinates. Exact scalar point previews remain available.",
        ),
      );
      return;
    }
    container.append(plotOptions(info, plotCoordinates));
  }
  function pointPreview(coordinates) {
    const preview = node("details", "coordinate-preview");
    preview.open = selection?.kind === "point";
    preview.append(node("summary", "", "Preview a checked case"));
    const pointForm = node("form", "coordinate-form");
    const pointInputs = new Map();
    coordinates.forEach((coordinate) => {
      const field = fieldsByPath.get(coordinate.path);
      const label = node("label", "range-label", coordinate.title);
      const input = node("input");
      input.type = "text";
      input.required = true;
      input.setAttribute("aria-label", `Preview coordinate ${coordinate.path}`);
      input.value =
        state.status === "checked"
          ? coordinate.point_value
          : (baseline?.fields.find((item) => item.path === coordinate.path)
              ?.value ?? "");
      label.append(input);
      pointInputs.set(coordinate.path, input);
      pointForm.append(label);
    });
    const previewAction = node("button", "primary", "Check this point");
    previewAction.type = "submit";
    pointForm.append(previewAction);
    pointForm.addEventListener("submit", (event) => {
      event.preventDefault();
      refresh(bindings, {
        point: Object.fromEntries(
          [...pointInputs].map(([path, input]) => [path, input.value]),
        ),
      });
    });
    preview.append(
      pointForm,
      node(
        "p",
        "input-help",
        "Previewing leaves your observations and filters unchanged.",
      ),
    );
    return preview;
  }
  function plotOptions(info, plotCoordinates) {
    const plot = node("details", "plot-options");
    plot.open =
      !chart.preferences &&
      (["curve", "region"].includes(info.kind) ||
        (info.kind === "section" && state.status === "checked"));
    plot.append(
      node(
        "summary",
        "",
        info.coordinate_count > 2
          ? "Take a section through the checked case"
          : "Explore the relationship",
      ),
    );
    const form = node("form", "plot-form");
    const controls = new Map();
    const viewLabel = node("label", "range-label", "Chart presentation");
    const view = node("select");
    view.setAttribute("aria-label", "Plot view");
    for (const [value, title] of [
      ["map", "Map"],
      ["surface", "3D surface"],
    ]) {
      const option = node("option", "", title);
      option.value = value;
      view.append(option);
    }
    view.value = chart.preferences?.view ?? "map";
    viewLabel.hidden =
      [...controls.values()].filter((value) => value.checkbox.checked).length <
      2;
    viewLabel.append(view);
    plotCoordinates.forEach((coordinate, index) => {
      const row = node("div", "plot-coordinate");
      const label = node("label", "plot-choice");
      const checkbox = node("input");
      checkbox.type = "checkbox";
      checkbox.checked = chart.preferences?.controls.some((path) =>
        plotCoordinates.some((item) => item.path === path),
      )
        ? chart.preferences.controls.includes(coordinate.path)
        : index < 2;
      checkbox.setAttribute("aria-label", `Vary ${coordinate.path}`);
      label.append(checkbox, node("span", "", coordinate.title));
      const windows = coordinate.window.map((initial, endpoint) => {
        const input = node("input");
        input.type = "text";
        input.value =
          chart.preferences?.windows[coordinate.path]?.[endpoint] ?? initial;
        input.placeholder = endpoint ? "Window maximum" : "Window minimum";
        input.setAttribute(
          "aria-label",
          `${endpoint ? "Upper" : "Lower"} plotting window for ${coordinate.path}`,
        );
        input.disabled = !checkbox.checked;
        return input;
      });
      checkbox.addEventListener("change", () => {
        if (
          [...controls.values()].filter((value) => value.checkbox.checked)
            .length > 2
        ) {
          checkbox.checked = false;
          get("plot-message").textContent =
            "Choose at most two coordinates. Uncheck a control before choosing another.";
        }
        windows.forEach((input) => (input.disabled = !checkbox.checked));
        viewLabel.hidden =
          [...controls.values()].filter((value) => value.checkbox.checked)
            .length < 2;
      });
      row.append(label, ...windows);
      controls.set(coordinate.path, { checkbox, windows });
      form.append(row);
    });
    viewLabel.hidden =
      [...controls.values()].filter((value) => value.checkbox.checked).length <
      2;
    const label = node("label", "range-label", "Follow this quantity");
    const output = node("select");
    output.setAttribute("aria-label", "Plot output");
    const choices = canonical();
    choices.forEach((field) => {
      const option = node("option", "", field.title);
      option.value = field.path;
      output.append(option);
    });
    output.value =
      chart.preferences?.output ??
      chart.data?.output ??
      (fieldsByPath.get(state.initial_field)?.depends_on.length
        ? state.initial_field
        : undefined) ??
      state.highlights.find(
        (path) => fieldsByPath.get(path)?.depends_on.length,
      ) ??
      choices.find((field) => field.kind === "derived")?.path ??
      choices[0]?.path;
    output.addEventListener("change", () => choose(output.value, false, false));
    label.append(output);

    const action = node("button", "primary", "Plot checked samples");
    action.type = "submit";
    const message = node("p", "input-help");
    message.id = "plot-message";
    message.setAttribute("role", "status");
    form.append(
      label,
      viewLabel,
      action,
      message,
      node(
        "p",
        "input-help",
        "Windows affect the picture only. Every plotted point is checked; gaps and missing samples do not prove infeasibility.",
      ),
    );
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      const selectedControls = [...controls].filter(
        ([, value]) => value.checkbox.checked,
      );
      if (!selectedControls.length) {
        message.textContent = "Choose at least one coordinate.";
        return;
      }
      loadPlot({
        controls: selectedControls.map(([path]) => path),
        windows: Object.fromEntries(
          selectedControls.map(([path, value]) => [
            path,
            value.windows.map((input) => input.value),
          ]),
        ),
        output: output.value,
        view: view.value,
      });
    });
    plot.append(form);
    return plot;
  }
  async function loadPlot(preferences) {
    const message = get("plot-message");
    if (!message) return;
    const action = get("family-controls").querySelector(
      ".plot-form button[type=submit]",
    );
    action.disabled = true;
    message.textContent = "Checking the sample points…";
    try {
      const reply = await chart.sample(preferences, {
        bindings,
        bounds,
        ...currentSelection(),
      });
      if (!reply) return;
      if (!reply.ok) {
        message.textContent = reply.result.error;
        return;
      }
      drawPlot();
      message.textContent = "";
    } catch (_) {
      message.textContent =
        "The plot could not be loaded. Your exploration is retained.";
    } finally {
      action.disabled = false;
    }
  }
  function drawPlot() {
    const container = get("family-plot");
    container.replaceChildren();
    if (!chart.data || state.raw) return;
    const data = chart.data;
    const points = data.points;
    container.append(
      node(
        "p",
        "plot-caption",
        `${data.section ? "Section · " : ""}${points.length} checked points · ${data.rejected} rejected · ${data.unknown} unresolved`,
      ),
    );
    if (data.section)
      container.append(
        node(
          "p",
          "input-help",
          "Held at the reference: " +
            Object.entries(data.fixed)
              .map(
                ([path, value]) =>
                  `${fieldsByPath.get(path).title} = ${scalarText(value, fieldsByPath.get(path))}`,
              )
              .join(" · "),
        ),
      );
    if (!points.length) {
      container.append(
        node(
          "p",
          "input-help",
          "No checked real points in this sample. Try another window or inspect the retained conditions.",
        ),
      );
      return;
    }
    familyCharts.render(
      container,
      data,
      fieldsByPath,
      chart.preferences?.view ?? "map",
      (point) => refresh(bindings, { point, fromPlot: true }),
      (view) => {
        chart.preferences = { ...chart.preferences, view };
        const input = get("family-controls").querySelector(
          '[aria-label="Plot view"]',
        );
        if (input) input.value = view;
        drawPlot();
      },
    );
  }
  function renderHeading(info) {
    const title = get("family-title"),
      summary = get("family-summary");
    const titles = {
      point: "A checked answer",
      cases: `${info?.case_count} checked alternatives`,
      empty: "No feasible answer",
      curve: "Explore one remaining coordinate",
      region: "Explore two remaining coordinates",
      section: "Explore a section of the family",
      finite: "Narrow the finite alternatives",
      conditional: "Some conditions remain unresolved",
      unsupported: "Model preview",
    };
    title.textContent = titles[info?.kind] ?? "Model preview";
    summary.textContent =
      info?.kind === "empty"
        ? "All retained observations, filters, and domains cannot hold together. Remove a bound or revise an observation to explore again."
        : info?.kind === "cases"
          ? "Compare the answers, preview one, or add an observation to distinguish them."
          : info?.kind === "unsupported"
            ? info.reason
            : `${info?.coordinate_count ?? 0} remaining coordinates · coordinate count does not establish the dimension of a coupled set. ${selection ? "Previewing a checked case; observations are unchanged." : "Select what you know and explore the possible values."}`;
  }
  function renderFilters(info) {
    const filters = get("family-filters");
    filters.replaceChildren();
    canonical()
      .filter((field) => field.bounds.some(Boolean))
      .forEach((field) => {
        filters.append(
          button(
            `${field.title}: ${field.bounds[0] || "…"} to ${field.bounds[1] || "…"} · remove`,
            () => removeBound(field),
            "bound-chip",
          ),
        );
      });
    if (info?.conditions?.length) {
      const retained = node("details", "retained-conditions");
      retained.append(
        node("summary", "", `${info.conditions.length} active filters`),
      );
      info.conditions.forEach((condition) =>
        retained.append(node("p", "set-exact", condition)),
      );
      filters.append(retained);
    }
  }
  function renderInsights(info) {
    const insights = get("family-insights");
    insights.replaceChildren();
    (info?.suggestions ?? []).forEach((path) => {
      const groups = new Set(
        info.cases.map((entry) => entry.values[path].value),
      ).size;
      insights.append(
        button(
          `An observation of ${fieldsByPath.get(path).title} ${groups === info.case_count ? "distinguishes" : "can narrow"} these cases`,
          () => choose(path),
        ),
      );
    });
    (info?.invariants ?? [])
      .filter((path) => state.highlights.includes(path))
      .slice(0, 3)
      .forEach((path) =>
        insights.append(
          node(
            "p",
            "input-help",
            `${fieldsByPath.get(path).title} ${info.cases?.length ? "is the same in every checked case" : "does not vary with the remaining coordinates"}.`,
          ),
        ),
      );
    (info?.relations ?? []).forEach((relation) =>
      insights.append(
        node(
          "p",
          "response-insight",
          `Each 1 ${fieldsByPath.get(relation.source).unit} change in ${fieldsByPath.get(relation.source).title} corresponds to a ${relation.approximate ? "≈" : ""}${relation.display_value} ${fieldsByPath.get(relation.output).unit} change in ${fieldsByPath.get(relation.output).title}.`,
        ),
      ),
    );
    if (info?.reason) insights.append(node("p", "input-help", info.reason));
    (info?.violations ?? [])
      .filter((path) => fieldsByPath.has(path))
      .forEach((path) =>
        insights.append(
          button(
            `Inspect the failing requirement on ${fieldsByPath.get(path).title}`,
            () => choose(path),
          ),
        ),
      );
    if (info?.kind === "finite")
      insights.append(
        node(
          "p",
          "input-help",
          `${info.candidate_count} coordinate proposals; automatic enumeration is limited to ${info.case_limit}. Narrow the family or preview a point.`,
        ),
      );
  }
  function render() {
    const root = get("family");
    root.hidden = state.raw;
    if (state.raw) return;
    const info = state.exploration;
    renderHeading(info);
    renderFilters(info);
    renderInsights(info);
    caseTable(info ?? {});
    coordinateForms(info ?? {});
    compare();
    drawPlot();
    const preset = chart.takePending();
    if (preset) loadPlot(preset);
  }
  new ResizeObserver(() => {
    if (chart.data) drawPlot();
  }).observe(get("family-plot"));
  return { request, accepted, render, rangeEditor, familyValues };
})();
