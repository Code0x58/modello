"use strict";
const $ = (id) => document.getElementById(id);
const kinds = {
  input: {
    label: "Input",
    explanation:
      "A supplied observation. Edit it or release it to explore alternatives.",
  },
  free: {
    label: "Free",
    explanation:
      "This input is open. Set a value or explore its symbolic relationships.",
  },
  derived: {
    label: "Calculated",
    explanation:
      "Calculated from the model; it updates when inputs change. You may observe it as a target.",
  },
  fixed: {
    label: "Recorded",
    explanation:
      "A retained component observation in this recorded problem; it cannot be edited here.",
  },
};
const domains = {
  positive: "Positive",
  nonnegative: "Zero or greater",
  negative: "Negative",
  nonpositive: "Zero or less",
  real: "Real number",
  integer: "Integer",
  rational: "Rational number",
  nonzero: "Non-zero",
};
let state,
  bindings = {},
  selected = "",
  busy = false,
  equationsVisible = false,
  activeScenario = null;
let fieldsByPath = new Map(),
  changed = new Set(),
  errors = new Set();
const drafts = new Map(),
  collapsed = new Set();
const node = (tag, className, text) => {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
};
function math(markup, inert = false) {
  const element = new DOMParser().parseFromString(
    markup,
    "application/xml",
  ).documentElement;
  if (element.localName !== "math")
    return node("span", "", "Expression unavailable");
  if (inert)
    element.querySelectorAll("[data-path]").forEach((symbol) => {
      ["data-path", "tabindex", "role", "aria-label"].forEach((name) =>
        symbol.removeAttribute(name),
      );
    });
  return document.importNode(element, true);
}
function badge(field) {
  const element = node("span", `badge ${field.kind}`, kinds[field.kind].label);
  element.title = kinds[field.kind].explanation;
  return element;
}
function value(field, large = false) {
  const element = node("div", `quantity-value${large ? " large" : ""}`);
  if (field.value_state === "hidden")
    element.append(node("span", "muted", field.value));
  else if (field.value_state === "free")
    element.append(node("span", "open-value", "Not specified"));
  else if (field.value_state === "numeric" && field.display_value) {
    if (field.approximate) {
      const approx = node("span", "approx", "≈");
      approx.title =
        "Rounded for display. The exact value is available in the inspector.";
      element.append(approx);
    }
    element.append(node("span", "number", field.display_value));
  } else {
    element.append(math(field.value_mathml));
    if (field.number_format === "percent")
      element.append(node("span", "value-unit", "%"));
  }
  if (field.unit && !["free", "hidden"].includes(field.value_state))
    element.append(node("span", "value-unit", field.unit));
  return element;
}
function choose(path, focus = false, scroll = true) {
  selected = path;
  details();
  document
    .querySelectorAll("[data-field]")
    .forEach((element) =>
      element.classList.toggle("selected", element.dataset.field === path),
    );
  if (scroll && matchMedia("(max-width: 800px)").matches)
    $("inspector").scrollIntoView({ behavior: "smooth", block: "start" });
  if (focus) $("binding-input")?.focus({ preventScroll: true });
}
function showError(message, paths = []) {
  errors = new Set(paths);
  $("message").replaceChildren(
    node("strong", "", "Change not applied"),
    node("span", "", message),
  );
  const names = paths
    .map((path) => fieldsByPath.get(path)?.title)
    .filter(Boolean);
  if (names.length)
    $("message").append(
      node("small", "", `Check: ${[...new Set(names)].join(" · ")}`),
    );
  $("message").hidden = false;
  document
    .querySelectorAll("[data-field]")
    .forEach((element) =>
      element.classList.toggle("error", errors.has(element.dataset.field)),
    );
  if ($("binding-input"))
    $("binding-input").setAttribute(
      "aria-invalid",
      String(errors.has(selected)),
    );
}
async function refresh(next = bindings, options = {}) {
  if (busy) return;
  const keepFocus = document.activeElement?.id === "binding-input";
  busy = true;
  $("canvas").setAttribute("aria-busy", "true");
  document.querySelectorAll("button, input").forEach((element) => {
    element.disabled = true;
  });
  try {
    const response = await fetch("/api/explore", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ bindings: next, raw: $("raw").checked }),
    });
    const data = await response.json();
    if (!response.ok) {
      showError(data.error || "Could not update the model.", data.paths || []);
      return;
    }
    changed = new Set(
      data.fields
        .filter(
          (field) =>
            fieldsByPath.has(field.path) &&
            fieldsByPath.get(field.path).value !== field.value,
        )
        .map((field) => field.path),
    );
    if (
      !state &&
      data.scenarios.length &&
      !Object.keys(data.scenarios[0].bindings).length
    )
      activeScenario = 0;
    bindings = { ...next };
    state = data;
    errors.clear();
    drafts.clear();
    if (options.select) selected = options.select;
    if (options.scenario !== undefined) activeScenario = options.scenario;
    else if (options.edit) activeScenario = null;
    render();
  } catch (_) {
    showError(
      "The local explorer could not be reached. Your last successful model is still shown.",
    );
  } finally {
    busy = false;
    $("canvas").setAttribute("aria-busy", "false");
    $("raw").checked = state?.raw || false;
    document.querySelectorAll("button, input").forEach((element) => {
      element.disabled = false;
    });
    if (keepFocus) $("binding-input")?.focus({ preventScroll: true });
  }
}
function apply(field, text) {
  const next = { ...bindings };
  field.aliases.forEach((alias) => delete next[alias]);
  next[field.path] = text.trim() || null;
  return refresh(next, { edit: true });
}
function section(title) {
  const element = node("section", "detail-section");
  element.append(node("h3", "detail-label", title));
  return element;
}
function fieldLink(path, title) {
  const link = node("button", "field-link", title);
  link.type = "button";
  link.addEventListener("click", () => choose(path, false, false));
  return link;
}
function details() {
  const field = fieldsByPath.get(selected);
  if (!field) return;
  $("field-title").textContent = field.title;
  $("field-symbol").replaceChildren(math(field.symbol_mathml, true));
  $("field-path").textContent = field.path.split(".").join("  /  ");
  $("field-description").textContent =
    field.description || "A quantity in this model.";
  const detail = $("field-detail");
  detail.replaceChildren();
  const current = node("div", "current-value");
  const source = node("div", "value-source");
  source.append(badge(field), node("span", "", field.provenance));
  current.append(value(field, true), source);
  detail.append(current);
  if (field.value_state === "symbolic" && field.depends_on.length) {
    const dependency = node("div", "dependency-note", "Still depends on ");
    field.depends_on.forEach((item) =>
      dependency.append(fieldLink(item.path, item.title)),
    );
    detail.append(dependency);
  }
  if (field.editable) {
    const form = node("form", "observation-form");
    const label = node(
      "label",
      "detail-label",
      field.kind === "input"
        ? "YOUR OBSERVATION"
        : field.kind === "free"
          ? "SET THIS INPUT"
          : "OBSERVE A TARGET",
    );
    label.htmlFor = "binding-input";
    const input = node("input");
    input.id = "binding-input";
    input.type = "text";
    input.setAttribute("aria-label", `Value for ${field.path}`);
    input.setAttribute("aria-invalid", String(errors.has(field.path)));
    input.setAttribute("aria-describedby", "input-help");
    input.value = drafts.has(field.path) ? drafts.get(field.path) : field.input;
    input.placeholder =
      field.number_format === "percent" ? "e.g. 70%" : "e.g. 3/2";
    input.addEventListener("input", () => drafts.set(field.path, input.value));
    const actions = node("div", "form-actions");
    actions.append(node("button", "primary", "Apply value"));
    if (field.kind === "input") {
      const release = node("button", "ghost", "Release input");
      release.type = "button";
      release.addEventListener("click", () => apply(field, ""));
      actions.append(release);
    }
    const hint = node(
      "p",
      "input-help",
      field.number_format === "percent"
        ? "Use 70% or 7/10. Clear the value to release the observation."
        : "Numbers and exact arithmetic, such as 3/2 or sqrt(2). Clear to release.",
    );
    hint.id = "input-help";
    form.append(label, input, actions, hint);
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      apply(field, input.value);
    });
    detail.append(form);
  } else
    detail.append(
      node(
        "p",
        "input-help",
        field.value_state === "hidden"
          ? "This background quantity stays in the model without an editor."
          : "This result is retained from a component snapshot.",
      ),
    );
  const domain = section("Domain & assumptions");
  const tags = node("div", "domain-tags");
  (field.assumptions.length ? field.assumptions : ["finite scalar"]).forEach(
    (name) =>
      tags.append(
        node("span", "domain-tag", domains[name] || name.replaceAll("_", " ")),
      ),
  );
  domain.append(tags);
  detail.append(domain);
  const own = field.constraints.find(
    (constraint) =>
      constraint.kind === "definition" &&
      field.aliases.includes(constraint.path),
  );
  if (own) {
    const definition = section("Model definition");
    const equation = node("div", "equation-card");
    equation.append(math(own.mathml));
    definition.append(equation);
    const expand = node("button", "equation-expand", "View in equations ↗");
    expand.type = "button";
    expand.addEventListener("click", () => {
      equationsVisible = true;
      setView();
      document
        .querySelector(`[data-equation="${CSS.escape(own.path)}"]`)
        ?.scrollIntoView({ behavior: "smooth", block: "center" });
    });
    definition.append(expand);
    detail.append(definition);
  }
  if (field.info) detail.append(node("p", "field-note", field.info));
  const related = [
    ...new Set(field.constraints.map((constraint) => constraint.path)),
  ].filter(
    (path) =>
      path !== field.path &&
      fieldsByPath.has(path) &&
      !field.aliases.includes(path),
  );
  if (related.length) {
    const connections = section("Connected quantities");
    const links = node("div", "connections");
    related.forEach((path) =>
      links.append(fieldLink(path, fieldsByPath.get(path).title)),
    );
    connections.append(links);
    detail.append(connections);
  }
  if (field.aliases.length > 1) {
    const aliases = section("Same quantity, other names");
    field.aliases
      .filter((path) => path !== field.path)
      .forEach((path) =>
        aliases.append(fieldLink(path, fieldsByPath.get(path)?.title || path)),
      );
    detail.append(aliases);
  }
  if (field.value_state === "numeric") {
    const exact = node("details", "exact-value");
    exact.append(node("summary", "", "Exact value"));
    const expression = node("div", "equation-card");
    expression.append(math(field.exact_mathml));
    if (field.number_format === "percent")
      expression.append(node("span", "value-unit", "%"));
    exact.append(expression);
    detail.append(exact);
  }
}
function quantity(field) {
  const row = node(
    "div",
    `quantity${changed.has(field.path) ? " updated" : ""}`,
  );
  row.dataset.field = field.path;
  const label = node("button", "row-select");
  label.type = "button";
  label.setAttribute("aria-label", `Inspect ${field.path}`);
  const icon = node("span", "row-symbol");
  icon.append(math(field.symbol_mathml, true));
  const name = node("span", "row-name");
  name.append(node("strong", "", field.title));
  if (field.unit) name.append(node("small", "", field.unit));
  label.append(icon, name);
  label.addEventListener("click", () => choose(field.path));
  const action = node(
    "button",
    field.kind === "free" ? "set-value" : "row-action",
    field.kind === "free"
      ? "Set value"
      : field.kind === "input"
        ? "Edit"
        : "Inspect",
  );
  action.type = "button";
  action.setAttribute("aria-label", `${action.textContent} ${field.path}`);
  action.addEventListener("click", () =>
    choose(field.path, field.kind === "free" || field.kind === "input"),
  );
  row.append(label, value(field), badge(field), action);
  return row;
}
function renderTree() {
  $("tree").replaceChildren();
  const components = new Map(
    state.components.map((component) => [component.path, component]),
  );
  const containers = new Map([["", $("tree")]]);
  function ensure(path) {
    if (containers.has(path)) return containers.get(path);
    const group = node("section", "component");
    const header = node("button", "component-head");
    header.type = "button";
    const info = components.get(path);
    header.append(
      node("span", "chevron", "⌄"),
      node("span", "", info?.title || path.split(".").at(-1)),
    );
    const count = state.fields.filter((field) =>
      field.path.startsWith(path + "."),
    ).length;
    header.append(node("span", "component-count", `${count} quantities`));
    header.title = info?.description || "Nested model";
    header.setAttribute("aria-expanded", String(!collapsed.has(path)));
    group.classList.toggle("collapsed", collapsed.has(path));
    header.addEventListener("click", () => {
      if (collapsed.has(path)) collapsed.delete(path);
      else collapsed.add(path);
      group.classList.toggle("collapsed", collapsed.has(path));
      header.setAttribute("aria-expanded", String(!collapsed.has(path)));
    });
    const body = node("div", "component-body");
    group.append(header, body);
    ensure(path.split(".").slice(0, -1).join(".")).append(group);
    containers.set(path, body);
    return body;
  }
  const fields = state.fields.filter(
    (field) => !state.highlights.includes(field.path),
  );
  fields
    .filter((field) => !field.depth && ["input", "free"].includes(field.kind))
    .forEach((field) => $("tree").append(quantity(field)));
  fields
    .filter((field) => field.depth)
    .forEach((field) =>
      ensure(field.path.split(".").slice(0, -1).join(".")).append(
        quantity(field),
      ),
    );
  state.components.forEach((component) => ensure(component.path));
  fields
    .filter((field) => !field.depth && !["input", "free"].includes(field.kind))
    .forEach((field) => $("tree").append(quantity(field)));
  if (!state.fields.length)
    $("tree").append(
      node("p", "empty", "This model has no visible quantities."),
    );
}
function renderHighlights() {
  $("highlights").replaceChildren();
  $("highlights").hidden = !state.highlights.length;
  state.highlights.forEach((path) => {
    const field = fieldsByPath.get(path);
    const card = node(
      "article",
      `highlight${changed.has(path) ? " updated" : ""}`,
    );
    card.dataset.field = path;
    const top = node("div", "highlight-head");
    const title = node("button", "highlight-select", field.title);
    title.type = "button";
    title.setAttribute("aria-label", `Inspect ${path}`);
    top.append(title, badge(field));
    card.append(top, value(field, true));
    card.append(
      node(
        "small",
        "highlight-note",
        field.value_state === "symbolic"
          ? "Depends on open inputs"
          : field.provenance,
      ),
    );
    card.addEventListener("click", () => choose(path));
    $("highlights").append(card);
  });
}
function renderScenarios() {
  $("scenarios").hidden = !state.scenarios.length;
  $("scenario-buttons").replaceChildren();
  state.scenarios.forEach((scenario, index) => {
    const button = node(
      "button",
      `scenario${index === activeScenario ? " active" : ""}`,
      scenario.title,
    );
    button.type = "button";
    button.title = `${scenario.description} Restores the initial model inputs.`;
    button.setAttribute("aria-pressed", String(index === activeScenario));
    button.addEventListener("click", () =>
      refresh(scenario.bindings, { select: scenario.select, scenario: index }),
    );
    $("scenario-buttons").append(button);
  });
  $("scenario-description").textContent =
    activeScenario === null
      ? "Custom exploration · edit a quantity or choose a question to start again."
      : state.scenarios[activeScenario]?.description || "";
}
function setView() {
  $("exploration").hidden = equationsVisible;
  $("equations").hidden = !equationsVisible;
  $("explore-tab").classList.toggle("active", !equationsVisible);
  $("equations-tab").classList.toggle("active", equationsVisible);
  $("explore-tab").setAttribute("aria-pressed", String(!equationsVisible));
  $("equations-tab").setAttribute("aria-pressed", String(equationsVisible));
  $("view-label").textContent = equationsVisible
    ? "THE ORIGINAL MODEL"
    : "MODEL STRUCTURE";
  $("view-title").textContent = equationsVisible
    ? "The equations behind the results"
    : "Inputs & relationships";
}
function render() {
  fieldsByPath = new Map(
    [...state.fields, ...state.symbols].map((field) => [field.path, field]),
  );
  if (!fieldsByPath.has(selected))
    selected = state.initial_field || state.fields[0]?.path;
  $("title").textContent = state.title;
  document.title = `${state.title} · modello`;
  $("description").textContent = state.description;
  $("model-name").textContent = state.name;
  $("model-info").textContent = state.info;
  $("about").hidden = !state.info;
  $("message").hidden = true;
  const status =
    state.status === "raw"
      ? "Raw definitions"
      : state.status === "checked"
        ? "Checked case"
        : "Exploring";
  $("status").textContent = status + " ⓘ";
  $("status").className = `status ${state.status}`;
  $("tip-status").textContent =
    state.status === "raw"
      ? "Definitions are shown without solving. Your recorded inputs are retained."
      : state.status === "checked"
        ? "This numerical case satisfies every recorded equation and declared domain."
        : "This is a symbolic or undecided preview. Open inputs and remaining domain conditions are not yet a checked numerical case.";
  $("summary").textContent =
    `${state.fields.length} quantities · ${state.equations.length} equations`;
  $("background").textContent = state.hidden_inputs
    ? `${state.hidden_inputs} background inputs retained`
    : "Exact model · readable results";
  $("condition").textContent =
    state.status === "conditional"
      ? "Symbolic preview · set what you know to explore a case."
      : state.condition;
  renderScenarios();
  renderHighlights();
  renderTree();
  $("equation-list").replaceChildren();
  state.equations.forEach((equation) => {
    const block = node("section", "equation");
    block.dataset.equation = equation.path;
    const field = fieldsByPath.get(equation.path);
    const label = node(
      "button",
      "equation-label",
      field?.title || equation.path,
    );
    label.type = "button";
    label.addEventListener("click", () => choose(equation.path));
    block.append(label, math(equation.mathml));
    $("equation-list").append(block);
  });
  if (!state.equations.length)
    $("equation-list").append(
      node(
        "p",
        "empty",
        "No defining equations yet. Set a free quantity in Explore.",
      ),
    );
  setView();
  choose(selected, false, false);
}
function hideTooltips() {
  document.querySelectorAll('[role="tooltip"]').forEach((tip) => {
    tip.hidden = true;
  });
}
document.querySelectorAll("[data-help]").forEach((help) => {
  const button = help.querySelector("button"),
    tip = help.querySelector('[role="tooltip"]');
  const show = () => {
    hideTooltips();
    tip.hidden = false;
    const rect = button.getBoundingClientRect();
    tip.style.left =
      Math.max(12, Math.min(rect.left, innerWidth - tip.offsetWidth - 12)) +
      "px";
    tip.style.top =
      Math.max(
        12,
        rect.bottom + tip.offsetHeight + 12 > innerHeight
          ? rect.top - tip.offsetHeight - 8
          : rect.bottom + 8,
      ) + "px";
  };
  help.addEventListener("pointerenter", show);
  help.addEventListener("focusin", show);
  button.addEventListener("click", show);
  help.addEventListener("pointerleave", () => {
    if (!help.contains(document.activeElement)) tip.hidden = true;
  });
  help.addEventListener("focusout", (event) => {
    if (!help.contains(event.relatedTarget)) tip.hidden = true;
  });
});
document.addEventListener("pointerdown", (event) => {
  if (!event.target.closest("[data-help]")) hideTooltips();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") hideTooltips();
});
window.addEventListener("resize", hideTooltips);
window.addEventListener("scroll", hideTooltips, true);
$("explore-tab").addEventListener("click", () => {
  equationsVisible = false;
  setView();
});
$("equations-tab").addEventListener("click", () => {
  equationsVisible = true;
  setView();
});
$("raw").addEventListener("change", () => refresh());
$("reset").addEventListener("click", () =>
  refresh(
    {},
    {
      scenario: state.scenarios.findIndex(
        (scenario) => !Object.keys(scenario.bindings).length,
      ),
      select: state.initial_field,
    },
  ),
);
$("back").addEventListener("click", () =>
  $("canvas").scrollIntoView({ behavior: "smooth", block: "start" }),
);
["pointerover", "focusin"].forEach((type) =>
  $("equation-list").addEventListener(type, (event) => {
    const symbol = event.target.closest("[data-path]");
    if (symbol) choose(symbol.dataset.path, false, false);
  }),
);
["canvas", "inspector"].forEach((id) => {
  $(id).addEventListener("click", (event) => {
    const symbol = event.target.closest("[data-path]");
    if (symbol) choose(symbol.dataset.path);
  });
  $(id).addEventListener("keydown", (event) => {
    const symbol = event.target.closest("[data-path]");
    if (symbol && ["Enter", " "].includes(event.key)) {
      event.preventDefault();
      choose(symbol.dataset.path);
    }
  });
});
refresh();
