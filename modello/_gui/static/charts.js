"use strict";
// Render audited samples; model arithmetic and geometry belong to their own types.
const familyCharts = (() => {
  const { SampleGrid, FlatProjection, SurfaceProjection } = chartGeometry;
  const camera = { rotation: -45, tilt: 28 };
  const make = (name, attributes = {}, text) => {
    const element = document.createElementNS(
      "http://www.w3.org/2000/svg",
      name,
    );
    for (const [key, value] of Object.entries(attributes))
      if (value !== undefined) element.setAttribute(key, value);
    if (text !== undefined) element.textContent = text;
    return element;
  };
  const tick = (value) =>
    value.toLocaleString(undefined, {
      maximumFractionDigits: 2,
      maximumSignificantDigits: 4,
    });
  const label = (field) =>
    field.title +
    (field.number_format === "percent"
      ? " (%)"
      : field.unit
        ? ` (${field.unit})`
        : "");
  const colour = (fraction) =>
    `hsl(${190 + 35 * fraction} 58% ${88 - 48 * fraction}%)`;

  class SampleChart {
    constructor(container, data, fields, view, onSelect, onView) {
      this.container = container;
      this.grid = new SampleGrid(data);
      this.fields = fields;
      this.view = view;
      this.onSelect = onSelect;
      this.onView = onView;
      this.surface = this.grid.joint && view === "surface";
      this.width = Math.max(260, container.getBoundingClientRect().width);
      this.height = this.surface
        ? Math.max(340, Math.min(480, this.width / 2))
        : 300;
      this.svg = make("svg", {
        viewBox: `0 0 ${this.width} ${this.height}`,
        class: "family-chart",
        role: "img",
        "aria-label": `${this.label(this.grid.output)} · ${this.grid.joint ? view : "curve"} of checked samples`,
      });
      this.readout = node("p", "plot-readout");
      this.readout.setAttribute("role", "status");
    }
    label(path) {
      return label(this.fields.get(path));
    }
    description(point) {
      return this.grid.paths
        .map((path, axis) => `${this.label(path)}: ${tick(point.axes[axis])}`)
        .join(" · ");
    }
    render() {
      if (this.grid.joint) this.showViews();
      if (this.surface) this.showCamera();
      this.draw();
      this.container.append(this.svg);
      if (this.grid.joint) this.showLegend();
      this.showPicker();
    }
    draw() {
      this.svg.replaceChildren();
      const projection = this.surface
        ? new SurfaceProjection(this.grid, camera, this.width, this.height)
        : new FlatProjection(this.grid, this.width, this.height);
      const labels = this.surface
        ? this.surfaceAxes(projection)
        : this.flatAxes(projection);
      if (this.grid.joint) this.drawCells(projection);
      else this.drawCurve(projection);
      this.drawSamples(projection);
      this.svg.append(...labels);
    }
    drawCells(projection) {
      for (const cell of this.grid.mesh(projection))
        this.svg.append(
          make("polygon", {
            points: cell.vertices.map(([x, y]) => `${x},${y}`).join(" "),
            fill: colour(cell.level),
            class: `sample-cell ${this.surface ? "surface-cell" : "map-cell"}`,
          }),
        );
    }
    drawCurve(projection) {
      const path = [...this.grid.segments()]
        .flatMap((segment) =>
          segment.map((point, index) => {
            const [x, y] = projection.project(point.axes);
            return `${index ? "L" : "M"}${x},${y}`;
          }),
        )
        .join(" ");
      this.svg.append(make("path", { d: path, class: "response-line" }));
    }
    drawSamples(projection) {
      for (const point of this.grid.points) {
        const [x, y] = projection.project(point.axes);
        const circle = make("circle", {
          cx: x,
          cy: y,
          r: this.grid.joint ? (this.surface ? 1.8 : 2.7) : 3.5,
          class: "sample-point",
        });
        if (this.grid.joint)
          circle.style.fill = colour(
            this.grid.outputRange.fraction(this.grid.value(point)),
          );
        circle.append(make("title", {}, this.description(point)));
        circle.addEventListener("pointerenter", () => {
          this.readout.textContent = this.description(point);
        });
        circle.addEventListener("click", () => this.onSelect(point.point));
        this.svg.append(circle);
      }
    }
    flatAxes(projection) {
      const { margin, horizontal, vertical } = projection;
      const x = (value) => projection.x(value),
        y = (value) => projection.y(value);
      this.svg.append(
        make("rect", { ...projection.frame, class: "chart-frame" }),
      );
      const ticks = this.width < 400 ? 3 : 5;
      for (let index = 0; index < ticks; index++) {
        const fraction = index / (ticks - 1),
          xv = horizontal.at(fraction),
          yv = vertical.at(fraction);
        this.svg.append(
          make("line", {
            x1: x(xv),
            x2: x(xv),
            y1: margin.top,
            y2: this.height - margin.bottom,
            class: "chart-grid",
          }),
        );
        this.svg.append(
          make("line", {
            x1: margin.left,
            x2: this.width - margin.right,
            y1: y(yv),
            y2: y(yv),
            class: "chart-grid",
          }),
        );
        this.svg.append(
          make(
            "text",
            {
              x: x(xv),
              y: this.height - margin.bottom + 21,
              "text-anchor":
                index === 0 ? "start" : index === ticks - 1 ? "end" : "middle",
            },
            tick(xv),
          ),
        );
        this.svg.append(
          make(
            "text",
            { x: margin.left - 8, y: y(yv) + 4, "text-anchor": "end" },
            tick(yv),
          ),
        );
      }
      return [
        make(
          "text",
          { x: margin.left, y: this.height - 12 },
          this.label(this.grid.controls[0]),
        ),
        make(
          "text",
          { x: margin.left, y: 14 },
          this.label(
            this.grid.joint ? this.grid.controls[1] : this.grid.output,
          ),
        ),
      ];
    }
    surfaceAxes(projection) {
      const labels = [];
      for (const axis of projection.axes) {
        const a = projection.project(axis.start),
          b = projection.project(axis.end);
        const side = axis.index === 2 ? -1 : Math.sign(b[0] - a[0]) || 1;
        const field = this.fields.get(this.grid.paths[axis.index]);
        const symbol =
          field.symbol.length <= 18
            ? field.symbol
            : ["x", "y", "output"][axis.index];
        this.svg.append(
          make("line", {
            x1: a[0],
            y1: a[1],
            x2: b[0],
            y2: b[1],
            class: "surface-axis",
          }),
        );
        for (const fraction of [0, 0.5, 1]) {
          const at = axis.at(fraction),
            point = projection.project(at);
          labels.push(
            make(
              "text",
              {
                x: point[0] + side * 8,
                y: point[1] + (axis.index === 2 ? 3 : 18),
                "text-anchor": side < 0 ? "end" : "start",
                class: "surface-axis-label",
              },
              tick(at[axis.index]),
            ),
          );
        }
        labels.push(
          make(
            "text",
            {
              x: b[0] + (axis.index === 2 ? 8 : 0),
              y: b[1] + (axis.index === 2 ? -8 : 34),
              "text-anchor": axis.index === 2 ? "start" : "middle",
              class: "surface-axis-label",
            },
            symbol,
          ),
        );
      }
      return labels;
    }
    showViews() {
      const views = node("div", "chart-views");
      views.setAttribute("role", "group");
      views.setAttribute("aria-label", "Chart view");
      for (const [view, title] of [
        ["map", "Map"],
        ["surface", "Surface"],
      ]) {
        const button = node("button", "ghost", title);
        button.type = "button";
        button.setAttribute("aria-pressed", String(this.view === view));
        button.addEventListener("click", () => this.onView(view));
        views.append(button);
      }
      this.container.append(views);
    }
    showCamera() {
      const rotation = node("div", "surface-rotation");
      for (const [key, title, min, max] of [
        ["rotation", "Rotate surface", -180, 180],
        ["tilt", "Surface tilt", 10, 75],
      ]) {
        const field = node("label", "range-label", title),
          input = node("input");
        input.type = "range";
        input.min = min;
        input.max = max;
        input.value = camera[key];
        input.setAttribute("aria-label", title);
        input.addEventListener("input", () => {
          camera[key] = Number(input.value);
          this.draw();
        });
        field.append(input);
        rotation.append(field);
      }
      this.container.append(rotation);
    }
    showLegend() {
      this.container.append(
        node(
          "p",
          "chart-axis-key",
          `${this.label(this.grid.controls[0])} × ${this.label(this.grid.controls[1])} → ${this.label(this.grid.output)}`,
        ),
      );
      const legend = node("div", "chart-colour-key"),
        gradient = node("span", "chart-gradient");
      gradient.style.background = `linear-gradient(to right, ${colour(0)}, ${colour(1)})`;
      legend.append(
        node("span", "", tick(this.grid.outputRange.lower)),
        gradient,
        node("span", "", tick(this.grid.outputRange.upper)),
        node("span", "", this.label(this.grid.output)),
      );
      this.container.append(
        legend,
        node(
          "p",
          "input-help",
          `${this.surface ? "Surface faces" : "Shaded cells"} interpolate checked corners. Boundaries are approximate; missing cells stay open.`,
        ),
      );
    }
    showPicker() {
      const select = node("select");
      select.setAttribute("aria-label", "Inspect a plotted sample");
      const placeholder = node(
        "option",
        "",
        "Choose a checked sample to inspect",
      );
      placeholder.value = "";
      select.append(placeholder);
      this.grid.points.forEach((point, index) => {
        const option = node("option", "", this.description(point));
        option.value = index;
        select.append(option);
      });
      select.addEventListener("change", () => {
        if (select.value !== "")
          this.onSelect(this.grid.points[Number(select.value)].point);
      });
      this.container.append(select, this.readout);
    }
  }
  return { render: (...arguments_) => new SampleChart(...arguments_).render() };
})();
