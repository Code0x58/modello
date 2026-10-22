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
  mixed: {
    label: "Mixed",
    explanation: "Entries have different sources. Inspect the vector to see which are observed, free, or calculated.",
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
  activeView = "explore",
  activeScenario = null;
let fieldsByPath = new Map(),
  changed = new Set(),
  errors = new Set();
const drafts = new Map(),
  inputEdits = new Map(),
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
  document.querySelectorAll(".input-row").forEach((row) => {
    row
      .querySelector(".case-value")
      .setAttribute("aria-invalid", String(errors.has(row.dataset.input)));
  });
}
async function refresh(next = bindings, options = {}) {
  if (busy) return;
  const keepFocus = document.activeElement?.id === "binding-input";
  busy = true;
  $("canvas").setAttribute("aria-busy", "true");
  const disabledBefore = new Map();
  document.querySelectorAll("button, input, textarea").forEach((element) => {
    disabledBefore.set(element, element.disabled);
    element.disabled = true;
  });
  const unlock = () => {
    disabledBefore.forEach((disabled, element) => {
      if (element.isConnected) element.disabled = disabled;
    });
    disabledBefore.clear();
  };
  try {
    const response = await fetch("/api/explore", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ bindings: next, raw: $("raw").checked }),
    });
    const data = await response.json();
    if (!response.ok) {
      showError(data.error || "Could not update the model.", data.paths || []);
      if (options.commitInputs)
        $("message").scrollIntoView({ behavior: "smooth", block: "start" });
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
    bindings = data.bindings ?? { ...next };
    state = data;
    errors.clear();
    drafts.clear();
    if (options.commitInputs || options.scenario !== undefined)
      inputEdits.clear();
    if (options.select) selected = options.select;
    if (options.scenario !== undefined) activeScenario = options.scenario;
    else if (options.edit) activeScenario = null;
    unlock();
    render();
  } catch (_) {
    showError(
      "The local explorer could not be reached. Your last successful model is still shown.",
    );
  } finally {
    busy = false;
    $("canvas").setAttribute("aria-busy", "false");
    $("raw").checked = state?.raw || false;
    unlock();
    if (keepFocus) $("binding-input")?.focus({ preventScroll: true });
  }
}
function withoutAliases(base, field) {
  const next = { ...base };
  (field.observation_paths || field.aliases).forEach((alias) => delete next[alias]);
  return next;
}
function apply(field, text) {
  const next = withoutAliases(bindings, field);
  next[field.path] = text.trim() || null;
  return refresh(next, { edit: true });
}
function restore(field) {
  return refresh(withoutAliases(bindings, field), { edit: true });
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
  if (
    field.editable &&
    field.initial_input &&
    field.input_state !== "initial"
  ) {
    const history = node("div", "input-history");
    history.append(node("p", "", `Original input: ${field.initial_input}.`));
    if (field.input_state === "dropped")
      history.append(
        node(
          "p",
          "",
          "You are no longer supplying this value. The equations still apply.",
        ),
      );
    const revert = node("button", "ghost", "Restore original input");
    revert.type = "button";
    revert.addEventListener("click", () => restore(field));
    history.append(revert);
    detail.append(history);
  }
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
      field.shape ? "YOUR VECTOR OBSERVATIONS" : field.kind === "input"
        ? "YOUR OBSERVATION"
        : field.kind === "free"
          ? "SET THIS INPUT"
          : "OBSERVE A TARGET",
    );
    label.htmlFor = "binding-input";
    const input = node(field.shape ? "textarea" : "input");
    input.id = "binding-input";
    if (field.shape) input.rows = 3;
    else input.type = "text";
    input.setAttribute("aria-label", `Value for ${field.path}`);
    input.setAttribute("aria-invalid", String(errors.has(field.path)));
    input.setAttribute("aria-describedby", "input-help");
    input.value = drafts.has(field.path) ? drafts.get(field.path) : field.input;
    input.placeholder =
      field.shape ? "e.g. [1, ?, 3/2]" : field.number_format === "percent" ? "e.g. 70%" : "e.g. 3/2";
    input.addEventListener("input", () => drafts.set(field.path, input.value));
    const actions = node("div", "form-actions");
    actions.append(node("button", "primary", field.shape ? "Apply entries" : "Apply value"));
    if (field.input) {
      const release = node("button", "ghost", field.shape ? "Drop observations" : "Drop input");
      release.type = "button";
      release.addEventListener("click", () => apply(field, ""));
      actions.append(release);
    }
    const hint = node(
      "p",
      "input-help",
      field.shape
        ? "Use brackets and comma-separated exact values, such as [1, ?, 3/2]. A ? leaves that entry unobserved. You can also edit entries individually below."
        : field.number_format === "percent"
        ? "Use 70% or 7/10. Clear to stop supplying a value; the equations still apply."
        : "Exact arithmetic, such as 3/2 or sqrt(2). Clear to stop supplying a value; the equations still apply.",
    );
    hint.id = "input-help";
    form.append(label, input, actions, hint);
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      apply(field, input.value);
    });
    detail.append(form);
    const manage = node("button", "manage-inputs", "Change inputs together ↗");
    manage.type = "button";
    manage.addEventListener("click", () => {
      activeView = "inputs";
      setView();
      $("canvas").scrollIntoView({ behavior: "smooth", block: "start" });
    });
    detail.append(manage);
  } else
    detail.append(
      node(
        "p",
        "input-help",
        field.shape
          ? "Select an entry to inspect it. Whole-vector editing is available when every entry can be edited."
          : field.value_state === "hidden"
          ? "This background quantity stays in the model without an editor."
          : "This result is retained from a component snapshot.",
      ),
    );
  if (field.shape) {
    const entries = section("Vector entries");
    const list = node("div", "vector-entries");
    field.entries.forEach((path) => {
      const entry = fieldsByPath.get(path);
      const row = node("div", "vector-entry");
      row.append(fieldLink(path, entry?.title || path), badge(entry));
      list.append(row);
    });
    entries.append(list);
    detail.append(entries);
  }
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
      activeView = "equations";
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
    [...state.components, ...(state.vectors || [])].map((component) => [component.path, component]),
  );
  const containers = new Map([["", $("tree")]]);
  function ensure(path) {
    if (containers.has(path)) return containers.get(path);
    const group = node("section", "component");
    const header = node("button", "component-head");
    header.type = "button";
    const info = components.get(path);
    if (info?.shape) {
      group.classList.add("vector");
      group.dataset.field = path;
    }
    header.append(node("span", "chevron", "⌄"));
    if (info?.shape) header.append(math(info.symbol_mathml, true));
    header.append(node("span", "", info?.title || path.split(".").at(-1)));
    const count = state.fields.filter((field) =>
      field.parent === path || field.path.startsWith(path + "."),
    ).length;
    header.append(node("span", "component-count", `${count} ${info?.shape ? "entries" : "quantities"}`));
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
    if (info?.shape) {
      const heading = node("div", "vector-heading");
      const inspect = node("button", "vector-inspect", "Inspect vector");
      inspect.type = "button";
      inspect.setAttribute("aria-label", `Inspect vector ${path}`);
      inspect.addEventListener("click", () => choose(path));
      heading.append(header, inspect);
      group.append(heading, body);
    } else group.append(header, body);
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
      ensure(field.parent ?? field.path.split(".").slice(0, -1).join(".")).append(
        quantity(field),
      ),
    );
  state.components.forEach((component) => ensure(component.path));
  (state.vectors || []).forEach((vector) => ensure(vector.path));
  fields
    .filter((field) => !field.depth && !["input", "free"].includes(field.kind))
    .forEach((field) => $("tree").append(quantity(field)));
  if (!state.fields.length && !(state.vectors || []).length)
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
function updateInputActions() {
  const count = inputEdits.size;
  $("inputs-tab").textContent = count ? `Inputs (${count})` : "Inputs";
  $("input-draft-status").textContent = count
    ? `${count} pending ${count === 1 ? "change" : "changes"}. Results still show the applied case.`
    : "No pending changes. Unchecked quantities are not supplied as inputs.";
  $("apply-inputs").disabled = !count;
  $("discard-inputs").disabled = !count;
  document.querySelector(".input-actions").classList.toggle("pending", !!count);
}
function renderInputs() {
  $("input-list").replaceChildren();
  const labels = {
    initial: "Original input",
    overridden: "Changed input",
    added: "Added input",
    dropped: "Dropped input",
    unspecified: "Not supplied",
  };
  const fields = state.fields.filter(
    (field) => field.editable && field.path === field.aliases[0],
  );
  fields.forEach((field, index) => {
    const edit = inputEdits.get(field.path);
    const row = node("div", "input-row");
    row.dataset.input = field.path;
    row.dataset.field = field.path;
    const supply = node("input", "input-supply");
    supply.type = "checkbox";
    supply.id = `supply-${index}`;
    supply.checked = edit ? edit.enabled : field.kind === "input";
    supply.setAttribute("aria-label", `Supply ${field.path}`);
    const name = node("label", "input-name");
    name.htmlFor = supply.id;
    const title = node("strong", "", field.title);
    if (field.unit)
      title.append(node("span", "input-unit", ` · ${field.unit}`));
    name.append(title);
    name.title = field.description;
    name.append(node("small", "input-path", field.path));
    if (field.initial_input)
      name.append(
        node("small", "input-original", `Original: ${field.initial_input}`),
      );
    const input = node("input", "case-value");
    input.type = "text";
    input.value = edit ? edit.text : field.input;
    input.setAttribute("aria-label", `Input value for ${field.path}`);
    input.setAttribute("aria-invalid", String(errors.has(field.path)));
    const source = node("span", "input-state");
    const revert = node("button", "input-restore", "↺");
    revert.type = "button";
    revert.title = "Restore the original input state in this draft";
    revert.setAttribute(
      "aria-label",
      `Restore original input for ${field.path}`,
    );
    const sync = () => {
      input.disabled = !supply.checked;
      input.required = supply.checked;
      input.placeholder = !supply.checked
        ? "Not supplied"
        : field.number_format === "percent"
          ? "e.g. 6%"
          : "Exact value";
      const pending = inputEdits.get(field.path);
      row.classList.toggle("pending", !!pending);
      source.textContent = pending
        ? pending.restore
          ? "Will restore"
          : pending.enabled
            ? "Will supply"
            : "Will drop"
        : labels[field.input_state];
      revert.disabled =
        !pending && ["initial", "unspecified"].includes(field.input_state);
      updateInputActions();
    };
    const record = () => {
      if (
        supply.checked
          ? field.kind === "input" && input.value.trim() === field.input
          : field.kind !== "input"
      )
        inputEdits.delete(field.path);
      else
        inputEdits.set(field.path, {
          enabled: supply.checked,
          text: input.value,
        });
      sync();
    };
    supply.addEventListener("change", () => {
      record();
      if (supply.checked) input.focus();
    });
    input.addEventListener("input", record);
    revert.addEventListener("click", () => {
      if (["initial", "unspecified"].includes(field.input_state))
        inputEdits.delete(field.path);
      else
        inputEdits.set(field.path, {
          enabled: !!field.initial_input,
          text: field.initial_input,
          restore: true,
        });
      supply.checked = !!field.initial_input;
      input.value = field.initial_input;
      sync();
    });
    row.append(supply, name, input, source, revert);
    $("input-list").append(row);
    sync();
  });
  if (!fields.length)
    $("input-list").append(
      node("p", "empty", "This model has no editable observations."),
    );
  updateInputActions();
}
function applyInputs() {
  let next = { ...bindings };
  for (const [path, edit] of inputEdits) {
    const field = fieldsByPath.get(path);
    next = withoutAliases(next, field);
    if (!edit.restore) next[path] = edit.enabled ? edit.text.trim() : null;
  }
  return refresh(next, { edit: true, commitInputs: true });
}
function setView() {
  $("exploration").hidden = activeView !== "explore";
  ["inputs", "equations"].forEach((view) => {
    $(view).hidden = activeView !== view;
  });
  ["explore", "inputs", "equations"].forEach((view) => {
    $(`${view}-tab`).classList.toggle("active", activeView === view);
    $(`${view}-tab`).setAttribute("aria-pressed", String(activeView === view));
  });
  $("view-label").textContent = {
    explore: "MODEL STRUCTURE",
    inputs: "OBSERVATIONS",
    equations: "THE ORIGINAL MODEL",
  }[activeView];
  $("view-title").textContent = {
    explore: "Inputs & relationships",
    inputs: "What are you supplying?",
    equations: "The equations behind the results",
  }[activeView];
}
function render() {
  fieldsByPath = new Map(
    [...state.fields, ...state.symbols, ...(state.vectors || [])].map((field) => [field.path, field]),
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
  renderInputs();
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
["explore", "inputs", "equations"].forEach((view) => {
  $(`${view}-tab`).addEventListener("click", () => {
    activeView = view;
    setView();
  });
});
$("inputs").addEventListener("submit", (event) => {
  event.preventDefault();
  applyInputs();
});
$("discard-inputs").addEventListener("click", () => {
  inputEdits.clear();
  errors.clear();
  render();
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
