// Run the page's script against a minimal DOM stub: catch runtime errors a
// syntax check cannot see, and prove the delegated click handlers change state.
const fs = require("fs");
const vm = require("vm");

const html = fs.readFileSync("index.html", "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function makeEl(id) {
  const el = {
    id,
    nodeType: 1,
    parentNode: null,
    _attrs: {},
    dataset: {},
    style: {},
    innerHTML: "",
    textContent: "",
    children: [],
    _listeners: {},
    addEventListener(type, fn) { this._listeners[type] = fn; },
    setAttribute(k, v) { this._attrs[k] = v; if (k === "data-stage") this.dataset.stage = v; },
    removeAttribute(k) { delete this._attrs[k]; },
    getAttribute(k) { return Object.prototype.hasOwnProperty.call(this._attrs, k) ? this._attrs[k] : null; },
    appendChild(c) { this.children.push(c); c.parentNode = this; return c; },
    getBoundingClientRect() { return { left: 0, top: 0, width: 760, height: 330 }; },
  };
  return el;
}

const cache = new Map();
const stageBtns = [0, 1, 2, 3].map((i) => {
  const el = makeEl("stage-" + i);
  el.setAttribute("data-stage", String(i));
  el.className = "stage-btn";
  return el;
});

const documentStub = {
  readyState: "complete",
  documentElement: makeEl("root"),
  _listeners: {},
  addEventListener(type, fn) { this._listeners[type] = fn; },
  getElementById(id) {
    if (!cache.has(id)) cache.set(id, makeEl(id));
    return cache.get(id);
  },
  querySelectorAll(sel) { return sel === ".stage-btn" ? stageBtns : []; },
  createElementNS() { return makeEl("svg-child"); },
};

const sandbox = {
  document: documentStub,
  window: { matchMedia: () => ({ matches: true, addEventListener() {} }) },
  localStorage: { getItem: () => null, setItem: () => {} },
  console,
};
sandbox.globalThis = sandbox;

function click(el) {
  const handler = documentStub._listeners.click;
  if (!handler) throw new Error("no delegated click handler on document");
  handler({ target: el });
}

try {
  vm.runInNewContext(script, sandbox, { filename: "page-script.js" });
  console.log("script ran without throwing");
  console.log("delegated click handler on document:", !!documentStub._listeners.click);

  click(documentStub.getElementById("themeBtn"));
  console.log("theme after click:", documentStub.documentElement.getAttribute("data-theme"));
  click(documentStub.getElementById("themeBtn"));
  console.log("theme after second click:", documentStub.documentElement.getAttribute("data-theme"));

  click(stageBtns[2]);
  console.log("aria-selected:", stageBtns.map((b) => b.getAttribute("aria-selected")).join(","));
  const detail = documentStub.getElementById("stageDetail");
  console.log("detail shows stage 3:", detail.innerHTML.indexOf("Train the conditional flow") >= 0);

  click(documentStub.getElementById("dirInv"));
  console.log("inverse pressed:", documentStub.getElementById("dirInv").getAttribute("aria-pressed"));
  console.log("loop opacity:", documentStub.getElementById("loopGroup").getAttribute("opacity"));
  console.log("caption mentions bisection:", documentStub.getElementById("flowCaption").innerHTML.indexOf("bisection") >= 0);

  console.log("chart children appended:", documentStub.getElementById("chart").children.length);
  console.log("launch rects drawn:", documentStub.getElementById("torchLaunches").children.length);
} catch (error) {
  console.log("THREW:", error && error.stack ? error.stack.split("\n").slice(0, 8).join("\n") : error);
}
