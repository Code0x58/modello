"use strict";
const $ = (id) => document.getElementById(id);
let state,
  bindings = {},
  selected = "",
  equationsVisible = false,
  busy = false;
const collapsed = new Set();
let fieldsByPath = new Map();
const node = (tag, className, text) => {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text !== undefined) el.textContent = text;
  return el;
};
function showError(text, paths = []) {
  $("message").textContent = text;
  $("message").hidden = false;
  document.querySelectorAll(".field").forEach((row) => {
    const invalid = paths.includes(row.dataset.path);
    row.classList.toggle("error", invalid);
    row.querySelector("input")?.setAttribute("aria-invalid", String(invalid));
  });
}
async function refresh(next = bindings) {
  if (busy) return;
  busy = true;
  document.querySelectorAll("button, input").forEach((el) => {
    el.disabled = true;
  });
  try {
    const response = await fetch("/api/explore", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ bindings: next, raw: $("raw").checked }),
    });
    const data = await response.json();
    if (!response.ok) {
      showError(data.error || "Could not update this model.", data.paths || []);
      $("raw").checked = state?.raw || false;
      return;
    }
    bindings = { ...next };
    state = data;
    render();
  } catch (_) {
    showError(
      "The local explorer could not be reached. Your last model is still shown.",
    );
  } finally {
    $("raw").checked = state?.raw || false;
    busy = false;
    document.querySelectorAll("button, input").forEach((el) => {
      el.disabled = false;
    });
  }
}
function details(path) {
  const field = fieldsByPath.get(path);
  if (!field) return;
  selected = path;
  document
    .querySelectorAll(".field")
    .forEach((row) =>
      row.classList.toggle("selected", row.dataset.path === path),
    );
  $("field-title").textContent = field.title;
  $("field-path").textContent = field.path;
  $("field-description").textContent =
    field.description || "A quantity in this model.";
  $("field-info").textContent = field.info;
  $("field-info").hidden = !field.info;
  const detail = $("field-detail");
  detail.replaceChildren();
  detail.append(
    node("div", "detail-label", "CURRENT VALUE"),
    node("div", "detail-value", field.value),
  );
  detail.append(
    node("div", "detail-label", "PROVENANCE"),
    node("span", `badge ${field.kind}`, field.kind),
    node("div", "detail-constraint", field.provenance),
  );
  detail.append(node("div", "detail-label", "DOMAIN & ASSUMPTIONS"));
  detail.append(
    node(
      "div",
      "detail-constraint",
      field.assumptions.join(" · ") || "Finite scalar",
    ),
  );
  detail.append(node("div", "detail-label", "RELATIONSHIPS"));
  field.constraints.forEach((c) =>
    detail.append(node("div", "detail-constraint", `${c.text} [${c.kind}]`)),
  );
  if (field.aliases.length > 1) {
    detail.append(
      node("div", "detail-label", "ALIASES"),
      node("div", "detail-constraint", field.aliases.join(" · ")),
    );
  }
}
function fieldRow(field) {
  const row = node("div", "field");
  row.dataset.path = field.path;
  const label = node("button", "field-label");
  label.type = "button";
  label.append(node("strong", "", field.title), node("small", "", field.path));
  label.addEventListener("click", () => details(field.path));
  row.append(label, node("span", `badge ${field.kind}`, field.kind));
  row.append(
    node(
      "div",
      `field-value ${["free", "derived"].includes(field.kind) ? "symbolic" : ""}`,
      field.value,
    ),
  );
  if (field.editable) {
    const editor = node("form", "input-editor");
    const input = node("input");
    input.type = "text";
    input.inputMode = "text";
    input.value = field.input;
    input.placeholder = "Bind value";
    input.setAttribute("aria-label", `Input ${field.path}`);
    input.title =
      "Exact arithmetic, e.g. 3/2 or sqrt(2). Clear to release. Press Enter to apply.";
    input.addEventListener("focus", () => details(field.path));
    editor.addEventListener("submit", (event) => {
      event.preventDefault();
      const next = { ...bindings };
      // Aliases share one observation, even when edited through another path.
      field.aliases.forEach((alias) => delete next[alias]);
      next[field.path] = input.value.trim() || null;
      refresh(next);
    });
    editor.append(input);
    row.append(editor);
  }
  const constraints = node("div", "constraint-preview");
  if (field.assumptions.length)
    constraints.append(node("span", "", field.assumptions.join(" · ")));
  const definition =
    field.constraints.find(
      (c) => c.kind === "definition" && c.path === field.path,
    ) || field.constraints.find((c) => c.kind === "definition");
  if (definition) constraints.append(node("span", "", definition.text));
  row.append(constraints);
  return row;
}
function renderTree() {
  $("tree").replaceChildren();
  const containers = new Map([["", $("tree")]]);
  const components = new Map(
    state.components.map((component) => [component.path, component]),
  );
  const ensure = (path) => {
    if (containers.has(path)) return containers.get(path);
    const parent = path.split(".").slice(0, -1).join(".");
    const group = node("section", "component");
    const info = components.get(path);
    const toggle = node(
      "button",
      "component-head",
      info?.title || path.split(".").at(-1),
    );
    toggle.type = "button";
    toggle.append(node("span", "", "component"));
    toggle.setAttribute("aria-expanded", String(!collapsed.has(path)));
    group.classList.toggle("collapsed", collapsed.has(path));
    toggle.addEventListener("click", () => {
      if (collapsed.has(path)) collapsed.delete(path);
      else collapsed.add(path);
      group.classList.toggle("collapsed", collapsed.has(path));
      toggle.setAttribute("aria-expanded", String(!collapsed.has(path)));
    });
    const body = node("div", "component-body");
    group.append(toggle, body);
    ensure(parent).append(group);
    containers.set(path, body);
    return body;
  };
  state.fields.forEach((field) =>
    ensure(field.path.split(".").slice(0, -1).join(".")).append(
      fieldRow(field),
    ),
  );
  state.components.forEach((component) => ensure(component.path));
  if (!state.fields.length)
    $("tree").append(
      node("p", "empty", "This model has no visible quantities."),
    );
}
function setView() {
  $("tree").hidden = equationsVisible;
  $("equations").hidden = !equationsVisible;
  $("explore-tab").classList.toggle("active", !equationsVisible);
  $("formula-tab").classList.toggle("active", equationsVisible);
  $("explore-tab").setAttribute("aria-pressed", String(!equationsVisible));
  $("formula-tab").setAttribute("aria-pressed", String(equationsVisible));
  $("view-label").textContent = equationsVisible
    ? "SIMULTANEOUS EQUATIONS"
    : "MODEL STRUCTURE";
  $("view-title").textContent = equationsVisible
    ? "The model, in its own symbols"
    : "Quantities & relationships";
}
function render() {
  fieldsByPath = new Map(
    [...state.fields, ...state.symbols].map((field) => [field.path, field]),
  );
  $("title").textContent = state.title;
  document.title = `${state.title} · modello`;
  $("model-name").textContent = state.name;
  $("description").textContent = state.description;
  $("model-info").textContent = state.info;
  $("model-info").hidden = !state.info;
  $("status").textContent =
    state.status === "raw"
      ? "○ Raw model"
      : state.status === "conditional"
        ? "◐ Conditional"
        : "● Checked model";
  $("status").classList.toggle("conditional", state.status === "conditional");
  $("summary").textContent =
    `${state.fields.length} quantities · ${state.equations.length} definitions${state.hidden_inputs ? ` · ${state.hidden_inputs} hidden inputs retained` : ""}`;
  $("condition").textContent = state.condition;
  $("message").hidden = true;
  renderTree();
  $("equation-list").replaceChildren();
  state.equations.forEach((equation, index) => {
    const block = node("div", "equation");
    block.append(
      node(
        "div",
        "equation-label",
        `${String(index + 1).padStart(2, "0")} / ${equation.path}`,
      ),
    );
    // Only server-generated MathML is inserted as markup. All labels use textContent.
    const math = new DOMParser().parseFromString(
      equation.mathml,
      "application/xml",
    ).documentElement;
    block.append(document.importNode(math, true));
    $("equation-list").append(block);
  });
  if (!state.equations.length)
    $("equation-list").append(
      node(
        "p",
        "empty",
        "No defining equations. Bind a free quantity in Explore model.",
      ),
    );
  setView();
  details(fieldsByPath.has(selected) ? selected : state.fields[0]?.path);
}
$("explore-tab").addEventListener("click", () => {
  equationsVisible = false;
  setView();
});
$("formula-tab").addEventListener("click", () => {
  equationsVisible = true;
  setView();
});
$("raw").addEventListener("change", () => refresh());
$("reset").addEventListener("click", () => refresh({}));
["pointerover", "focusin", "click"].forEach((type) =>
  $("equation-list").addEventListener(type, (event) => {
    const symbol = event.target.closest("[data-path]");
    if (symbol) details(symbol.dataset.path);
  }),
);
$("equation-list").addEventListener("keydown", (event) => {
  if (["Enter", " "].includes(event.key) && event.target.dataset.path) {
    event.preventDefault();
    details(event.target.dataset.path);
  }
});
refresh();
