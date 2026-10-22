"use strict";
// A selected point and a pinned comparison never become model observations.
const familyUI = (() => {
  let bounds = {},
    selection = null,
    baseline = null,
    plotData = null,
    plotPreferences = null,
    revision = 0;
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
    revision++;
    bounds = options.bounds ?? bounds;
    selection = data.exploration?.selection ?? null;
    if (
      !options.fromPlot &&
      (options.edit ||
        options.scenario !== undefined ||
        options.bounds ||
        options.point ||
        options.case !== undefined ||
        data.raw)
    )
      plotData = null;
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
    container.append(preview);
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
    const plot = node("details", "plot-options");
    plot.open =
      ["curve", "region"].includes(info.kind) ||
      Boolean(plotData) ||
      (info.kind === "section" && state.status === "checked");
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
    plotCoordinates.forEach((coordinate, index) => {
      const row = node("div", "plot-coordinate");
      const label = node("label", "plot-choice");
      const checkbox = node("input");
      checkbox.type = "checkbox";
      checkbox.checked = plotPreferences?.controls.some((path) =>
        plotCoordinates.some((item) => item.path === path),
      )
        ? plotPreferences.controls.includes(coordinate.path)
        : index < 2;
      checkbox.setAttribute("aria-label", `Vary ${coordinate.path}`);
      label.append(checkbox, node("span", "", coordinate.title));
      const windows = coordinate.window.map((initial, endpoint) => {
        const input = node("input");
        input.type = "text";
        input.value =
          plotPreferences?.windows[coordinate.path]?.[endpoint] ?? initial;
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
      });
      row.append(label, ...windows);
      controls.set(coordinate.path, { checkbox, windows });
      form.append(row);
    });
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
      plotData?.output ??
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
      action,
      message,
      node(
        "p",
        "input-help",
        "Windows affect the picture only. Every plotted point is checked; gaps and missing samples do not prove infeasibility.",
      ),
    );
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const selectedControls = [...controls].filter(
        ([, value]) => value.checkbox.checked,
      );
      if (!selectedControls.length) {
        message.textContent = "Choose at least one coordinate.";
        return;
      }
      action.disabled = true;
      const requestedRevision = revision;
      message.textContent = "Checking the sample points…";
      plotPreferences = {
        controls: selectedControls.map(([path]) => path),
        windows: Object.fromEntries(
          selectedControls.map(([path, value]) => [
            path,
            value.windows.map((input) => input.value),
          ]),
        ),
      };
      try {
        const response = await fetch("/api/plot", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            bindings,
            bounds,
            ...currentSelection(),
            ...plotPreferences,
            output: output.value,
          }),
        });
        const result = await response.json();
        if (requestedRevision !== revision) return;
        if (!response.ok) {
          message.textContent = result.error;
          return;
        }
        plotData = result;
        drawPlot();
        message.textContent = "";
      } catch (_) {
        message.textContent =
          "The plot could not be loaded. Your exploration is retained.";
      } finally {
        action.disabled = false;
      }
    });
    plot.append(form);
    container.append(plot);
  }
  function drawPlot() {
    const container = get("family-plot");
    container.replaceChildren();
    if (!plotData || state.raw) return;
    const data = plotData;
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
    const width = Math.max(260, container.getBoundingClientRect().width),
      height = 270;
    const margin = { left: 65, right: 24, top: 20, bottom: 65 };
    const make = (name, attributes = {}, text) => {
      const element = document.createElementNS(
        "http://www.w3.org/2000/svg",
        name,
      );
      for (const [key, value] of Object.entries(attributes))
        element.setAttribute(key, value);
      if (text !== undefined) element.textContent = text;
      return element;
    };
    const two = data.controls.length === 2,
      outputIndex = data.controls.length;
    const yValues = points.map((point) => point.axes[two ? 1 : outputIndex]);
    let minimum = Math.min(...yValues),
      maximum = Math.max(...yValues);
    const padding =
      maximum === minimum
        ? Math.max(Math.abs(minimum) * 0.05, 1)
        : (maximum - minimum) * 0.08;
    const xDomain = data.windows[0],
      yDomain = two ? data.windows[1] : [minimum - padding, maximum + padding];
    const x = (value) =>
      margin.left +
      ((value - xDomain[0]) / (xDomain[1] - xDomain[0])) *
        (width - margin.left - margin.right);
    const y = (value) =>
      height -
      margin.bottom -
      ((value - yDomain[0]) / (yDomain[1] - yDomain[0])) *
        (height - margin.top - margin.bottom);
    const svg = make("svg", {
      viewBox: `0 0 ${width} ${height}`,
      class: "family-chart",
      role: "img",
      "aria-label": `${fieldsByPath.get(data.output).title} across checked ${data.section ? "section" : "family"} samples`,
    });
    svg.append(
      make("rect", {
        x: margin.left,
        y: margin.top,
        width: width - margin.left - margin.right,
        height: height - margin.top - margin.bottom,
        class: "chart-frame",
      }),
    );
    const tickText = (value) =>
      value.toLocaleString(undefined, {
        maximumFractionDigits: 2,
        maximumSignificantDigits: 4,
      });
    const tickCount = width < 400 ? 3 : 4;
    for (let i = 0; i < tickCount; i++) {
      const fraction = i / (tickCount - 1),
        xv = xDomain[0] + fraction * (xDomain[1] - xDomain[0]),
        yv = yDomain[0] + fraction * (yDomain[1] - yDomain[0]);
      svg.append(
        make(
          "text",
          {
            x: x(xv),
            y: height - margin.bottom + 22,
            "text-anchor":
              i === 0 ? "start" : i === tickCount - 1 ? "end" : "middle",
          },
          tickText(xv),
        ),
      );
      svg.append(
        make(
          "text",
          { x: margin.left - 9, y: y(yv) + 4, "text-anchor": "end" },
          tickText(yv),
        ),
      );
    }
    const axisLabel = (path) => {
      const field = fieldsByPath.get(path);
      return (
        field.title +
        (field.number_format === "percent"
          ? " (%)"
          : field.unit
            ? ` (${field.unit})`
            : "")
      );
    };
    svg.append(
      make(
        "text",
        { x: margin.left, y: height - 14, "text-anchor": "start" },
        axisLabel(data.controls[0]),
      ),
    );
    svg.append(
      make(
        "text",
        { x: margin.left, y: 12, "text-anchor": "start" },
        axisLabel(two ? data.controls[1] : data.output),
      ),
    );
    if (!two) {
      let previous = -2;
      const path = points
        .map((point) => {
          const command = point.index === previous + 1 ? "L" : "M";
          previous = point.index;
          return `${command}${x(point.axes[0])},${y(point.axes[outputIndex])}`;
        })
        .join(" ");
      svg.append(make("path", { d: path, class: "response-line" }));
    }
    const outputValues = points.map((point) => point.axes[outputIndex]);
    const lower = Math.min(...outputValues),
      upper = Math.max(...outputValues);
    const readout = node("p", "plot-readout");
    readout.setAttribute("role", "status");
    const sampleSelect = node("select");
    sampleSelect.setAttribute("aria-label", "Inspect a plotted sample");
    const placeholder = node(
      "option",
      "",
      "Choose a checked sample to inspect",
    );
    placeholder.value = "";
    sampleSelect.append(placeholder);
    points.forEach((point, index) => {
      const coordinates = data.controls
        .map(
          (path, axis) =>
            `${fieldsByPath.get(path).title}: ${tickText(point.axes[axis])}`,
        )
        .join(" · ");
      const text = `${coordinates} · ${fieldsByPath.get(data.output).title}: ${tickText(point.axes[outputIndex])}`;
      const circle = make("circle", {
        cx: x(point.axes[0]),
        cy: y(point.axes[two ? 1 : outputIndex]),
        r: two ? 5 : 3.5,
        class: "sample-point",
      });
      if (two)
        circle.style.opacity = String(
          upper === lower
            ? 1
            : 0.25 +
                (0.75 * (point.axes[outputIndex] - lower)) / (upper - lower),
        );
      circle.append(make("title", {}, text));
      circle.addEventListener("pointerenter", () => {
        readout.textContent = text;
      });
      circle.addEventListener("click", () =>
        refresh(bindings, { point: point.point, fromPlot: true }),
      );
      svg.append(circle);
      const option = node("option", "", text);
      option.value = index;
      sampleSelect.append(option);
    });
    sampleSelect.addEventListener("change", () => {
      if (sampleSelect.value !== "")
        refresh(bindings, {
          point: points[Number(sampleSelect.value)].point,
          fromPlot: true,
        });
    });
    container.append(svg);
    if (two)
      container.append(
        node(
          "p",
          "input-help",
          `${fieldsByPath.get(data.output).title}: ${tickText(lower)} to ${tickText(upper)} · darker points indicate higher values`,
        ),
      );
    container.append(sampleSelect, readout);
  }
  function render() {
    const root = get("family");
    root.hidden = state.raw;
    if (state.raw) return;
    const info = state.exploration;
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
    caseTable(info ?? {});
    coordinateForms(info ?? {});
    compare();
    drawPlot();
  }
  new ResizeObserver(() => {
    if (plotData) drawPlot();
  }).observe(get("family-plot"));
  return { request, accepted, render, rangeEditor, familyValues };
})();
