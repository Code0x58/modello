"use strict";
// Exercise proposal state against real API fixtures without requiring a browser.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const source = fs.readFileSync(process.argv[2], "utf8");
const fixtures = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));

class Element {
  constructor() {
    this.disabled = false;
    this.checked = false;
    this.isConnected = true;
    this.dataset = {};
    this.classList = { toggle() {}, add() {}, remove() {} };
  }
  addEventListener() {}
  setAttribute() {}
  append() {}
  replaceChildren() {}
  focus() {}
  scrollIntoView() {}
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
    createElement() { return new Element(); },
    addEventListener() {},
  },
  window: { addEventListener() {} },
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
vm.runInContext("busy = false; render = () => {};", context);

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
  console.log("Vector edits, normalization, restore, and rejection passed");
})().catch(error => { console.error(error); process.exitCode = 1; });
