"use strict";
// Exercise proposal state against real API fixtures without requiring a browser.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const source = fs.readFileSync(process.argv[2], "utf8");
const fixtures = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
const created = [], queries = [], scrolled = [];

class Element {
  constructor() {
    this.disabled = false;
    this.checked = false;
    this.isConnected = true;
    this.dataset = {};
    this.children = [];
    this.classList = { toggle() {}, add() {}, remove() {} };
    this.listeners = new Map();
    created.push(this);
  }
  addEventListener(name, listener) { this.listeners.set(name, listener); }
  setAttribute() {}
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = [...children]; }
  focus() {}
  querySelectorAll() { return []; }
  scrollIntoView() { scrolled.push(this); }
}

const elements = new Map();
const textarea = new Element();
const requests = [];
const responses = [];
let booting = true;
const context = vm.createContext({
  document: {
    activeElement: null,
    getElementById(id) {
      if (!elements.has(id)) elements.set(id, new Element());
      return elements.get(id);
    },
    querySelectorAll(selector) {
      return selector.includes("textarea") ? [textarea] : [];
    },
    createElement(tag) { return Object.assign(new Element(), { tagName: tag }); },
    importNode(element) { return element; },
    querySelector(selector) { queries.push(selector); return new Element(); },
    addEventListener() {},
  },
  window: { addEventListener() {} },
  DOMParser: class {
    parseFromString() { return { documentElement: Object.assign(new Element(), { localName: "math" }) }; }
  },
  CSS: { escape: value => value },
  fetch: async (_, options) => {
    // Hold the initial request while the isolated test takes control.
    if (booting) return new Promise(() => {});
    assert.equal(textarea.disabled, true, "Vector editing must lock during a request");
    requests.push(JSON.parse(options.body));
    const response = responses.shift();
    assert.ok(response, "Unexpected request");
    return { ok: response.ok, json: async () => response.data };
  },
});
vm.runInContext(source, context);
booting = false;
textarea.disabled = false;
vm.runInContext(`busy = false; render = () => {
  fieldsByPath = new Map([...state.fields, ...(state.vectors || [])].map(field => [field.path, field]));
};`, context);

const evaluate = (code) => vm.runInContext(`(async () => { ${code} })()`, context);
const bindings = () => JSON.parse(vm.runInContext("JSON.stringify(bindings)", context));
const queue = (data, ok = true) => responses.push({ data, ok });

(async () => {
  queue(fixtures.initial);
  await evaluate("await refresh();");
  assert.deepEqual(bindings(), {});
  queue(fixtures.bulk);
  await evaluate("await refresh({samples: '[1, ?, 3]', total: '6'}, {edit: true});");
  assert.deepEqual(bindings(), fixtures.bulk.bindings);
  assert.ok(!Object.hasOwn(bindings(), "samples"), "Vector edits must normalize to coordinates");
  assert.equal(vm.runInContext("changed.has('samples')", context), true, "Vector highlights must reflect changed entries");
  assert.equal(vm.runInContext("changed.has('samples[1]')", context), true);
  assert.equal(textarea.disabled, false);
  queue(fixtures.edited);
  await evaluate("await apply(state.fields.find(field => field.path === 'samples[1]'), '2');");
  assert.deepEqual(requests.at(-1).bindings, fixtures.edited.bindings);
  assert.equal(bindings()["samples[1]"], "2");
  queue(fixtures.restored);
  await evaluate("await restore(state.vectors.find(field => field.path === 'samples'));");
  assert.deepEqual(requests.at(-1).bindings, { total: "6" });
  assert.deepEqual(bindings(), { total: "6" }, "Restoring the vector must retain an independent target");
  queue(fixtures.rejected, false);
  await evaluate("await apply(state.fields.find(field => field.path === 'samples[1]'), '4');");
  assert.deepEqual(bindings(), { total: "6" }, "Rejected edits must preserve the applied proposal");
  assert.equal(vm.runInContext("state.fields.find(field => field.path === 'total').value", context), "6");
  assert.equal(textarea.disabled, false);
  assert.equal(responses.length, 0);
  await evaluate("choose('doubled[0]', false, false);");
  const showEquation = created.filter(element => element.className === "equation-expand").at(-1);
  assert.ok(showEquation, "A defined vector entry must offer its original equation");
  showEquation.listeners.get("click")();
  assert.equal(vm.runInContext("activeView", context), "equations");
  assert.equal(queries.at(-1), '[data-equation="doubled"]');
  assert.ok(fixtures.restored.equations.some(equation => equation.path === "doubled"));
  assert.ok(scrolled.length, "Equation navigation must scroll to the complete vector definition");
  context.lawFixture = fixtures.laws;
  await evaluate("state = lawFixture; render(); renderEquations(); choose('closing', false, false);");
  const lawLabel = created.filter(element => element.className === "equation-label" && element.textContent === "Relative recurrence and boundary").at(-1);
  assert.ok(lawLabel, "A relative law must have a readable equation heading");
  assert.equal(lawLabel.tagName, "span");
  assert.equal(lawLabel.listeners.has("click"), false, "A law heading must not select a nonexistent scalar field");
  const showLaw = created.filter(element => element.className === "equation-expand").at(-1);
  showLaw.listeners.get("click")();
  assert.equal(queries.at(-1), '[data-equation="law"]');
  queue(fixtures.longer);
  await evaluate("await apply(state.vectors.find(field => field.path === 'levels'), '[2, 3, 5, 8, 13, 21, 34]');");
  assert.deepEqual(bindings(), fixtures.longer.bindings);
  assert.equal(vm.runInContext("state.fields.find(field => field.path === 'N').value", context), "6");
  queue(fixtures.entry_edit);
  await evaluate("await apply(state.fields.find(field => field.path === 'levels[6]'), '35');");
  assert.deepEqual(requests.at(-1).bindings, fixtures.entry_edit.bindings);
  assert.equal(Object.hasOwn(bindings(), "levels[6]"), false, "Entry edits must retain the vector's length");
  queue(fixtures.entry_restore);
  await evaluate("await restore(state.fields.find(field => field.path === 'levels[6]'));");
  assert.deepEqual(bindings(), fixtures.entry_restore.bindings);
  queue(fixtures.laws);
  await evaluate("await restore(state.vectors.find(field => field.path === 'levels'));");
  assert.deepEqual(requests.at(-1).bindings, {});
  assert.equal(vm.runInContext("state.fields.find(field => field.path === 'N').value", context), "4");
  context.sequenceFixture = fixtures.sequence;
  await evaluate("state = sequenceFixture; renderTree();");
  const groups = created.filter(element => element.dataset.component);
  const group = path => groups.findLast(element => element.dataset.component === path);
  const body = path => group(path).children.at(-1);
  assert.ok(body("fills").children.includes(group("fills[1]")), "Model-vector members belong inside the vector");
  assert.ok(body("states").children.includes(group("states[2]")), "State views use their declared parent");
  assert.equal(group("states.steps"), undefined, "Indexed recurrences need no transition components");
  assert.equal(fixtures.sequence.vectors.length, 0, "The demo should expose model vectors rather than column projections");
  await evaluate("choose('fills', false, false);");
  assert.equal(elements.get("field-title").textContent, "Executions");
  assert.ok(created.some(element => element.className === "model-member-value"));
  const fills = () => vm.runInContext("state.components.find(item => item.path === 'fills')", context);
  queue(fixtures.sequence_added);
  context.collectionFixture = fills();
  await evaluate("await resizeCollection(collectionFixture);");
  assert.equal(JSON.parse(requests.at(-1).bindings.fills).length, 3);
  assert.equal(requests.at(-1).bindings.N, null, "A resized collection determines its own horizon");
  queue(fixtures.sequence_cell);
  await evaluate("await apply(state.fields.find(field => field.path === 'fills[2].price'), '110');");
  assert.equal(JSON.parse(requests.at(-1).bindings.fills)[2].price, "110");
  assert.equal(Object.hasOwn(requests.at(-1).bindings, "fills[2].price"), false, "Cell edits must update the model record");
  context.sequenceFixture = fixtures.sequence;
  await evaluate("state = sequenceFixture; bindings = {};");
  queue(fixtures.sequence_removed);
  context.collectionFixture = fills();
  await evaluate("await resizeCollection(collectionFixture, 1);");
  assert.equal(JSON.parse(requests.at(-1).bindings.fills).length, 1);
  assert.equal(vm.runInContext("state.fields.find(field => field.path === 'terminal.cash').value", context), "799");
  queue(fixtures.sequence_empty);
  context.collectionFixture = fills();
  await evaluate("await resizeCollection(collectionFixture, 0);");
  assert.equal(JSON.parse(requests.at(-1).bindings.fills).length, 0);
  assert.equal(vm.runInContext("state.fields.find(field => field.path === 'terminal.cash').value", context), "1000");
  console.log("Vector proposals, restore, hierarchy, and model record editing passed");
})().catch(error => { console.error(error); process.exitCode = 1; });
