"use strict";
// Charts are views of audited samples. Interpolated faces are never set proofs.
const familyCharts = (() => {
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
  const span = (domain) => domain[1] - domain[0];

  function render(container, data, fields, view, onSelect, onView) {
    const points = data.points,
      two = data.controls.length === 2;
    if (!points.length) return;
    const outputIndex = data.controls.length;
    const values = points.map((point) => point.axes[outputIndex]);
    const lower = Math.min(...values),
      upper = Math.max(...values);
    const fraction = (value) =>
      upper === lower ? 0.5 : (value - lower) / (upper - lower);
    const width = Math.max(260, container.getBoundingClientRect().width);
    const height =
      two && view === "surface" ? Math.max(340, Math.min(480, width / 2)) : 300;
    const readout = node("p", "plot-readout");
    readout.setAttribute("role", "status");
    const text = (point) =>
      [...data.controls, data.output]
        .map(
          (path, index) =>
            `${label(fields.get(path))}: ${tick(point.axes[index])}`,
        )
        .join(" · ");
    const inspect = (element, point) => {
      element.append(make("title", {}, text(point)));
      element.addEventListener("pointerenter", () => {
        readout.textContent = text(point);
      });
      element.addEventListener("click", () => onSelect(point.point));
    };
    if (two) {
      const views = node("div", "chart-views");
      views.setAttribute("role", "group");
      views.setAttribute("aria-label", "Chart view");
      for (const [mode, title] of [
        ["map", "Map"],
        ["surface", "Surface"],
      ]) {
        const button = node("button", "ghost", title);
        button.type = "button";
        button.setAttribute("aria-pressed", String(view === mode));
        button.addEventListener("click", () => onView(mode));
        views.append(button);
      }
      container.append(views);
    }
    const svg = make("svg", {
      viewBox: `0 0 ${width} ${height}`,
      class: "family-chart",
      role: "img",
      "aria-label": `${label(fields.get(data.output))} · ${two ? view : "curve"} of checked samples`,
    });
    const byIndex = new Map(points.map((point) => [point.index, point]));
    const cells = [];
    if (two) {
      const size = data.grid_size;
      for (let i = 0; i < size - 1; i++) {
        for (let j = 0; j < size - 1; j++) {
          const corners = [
            i * size + j,
            (i + 1) * size + j,
            (i + 1) * size + j + 1,
            i * size + j + 1,
          ].map((index) => byIndex.get(index));
          // Never bridge rejected, unresolved, or absent corners.
          if (corners.every(Boolean)) cells.push(corners);
        }
      }
    }
    function faces(project, sort = false) {
      const mesh = cells.map((corners) => ({
        corners,
        projected: corners.map((point) => project(point.axes)),
        value:
          corners.reduce(
            (sum, point) => sum + fraction(point.axes[outputIndex]),
            0,
          ) / 4,
      }));
      if (sort)
        mesh.sort(
          (a, b) =>
            a.projected.reduce((sum, p) => sum + p[2], 0) -
            b.projected.reduce((sum, p) => sum + p[2], 0),
        );
      for (const cell of mesh) {
        svg.append(
          make("polygon", {
            points: cell.projected.map((p) => `${p[0]},${p[1]}`).join(" "),
            fill: colour(cell.value),
            class: `sample-cell ${sort ? "surface-cell" : "map-cell"}`,
          }),
        );
      }
    }
    function dots(project) {
      for (const point of points) {
        const p = project(point.axes);
        const circle = make("circle", {
          cx: p[0],
          cy: p[1],
          r: two ? (view === "surface" ? 1.8 : 2.7) : 3.5,
          class: "sample-point",
        });
        if (two) circle.style.fill = colour(fraction(point.axes[outputIndex]));
        inspect(circle, point);
        svg.append(circle);
      }
    }
    function flat() {
      const margin = { left: 58, right: 22, top: 25, bottom: 60 };
      const pad =
        upper === lower
          ? Math.max(Math.abs(lower) * 0.05, 1)
          : (upper - lower) * 0.08;
      const xDomain = data.windows[0],
        yDomain = two ? data.windows[1] : [lower - pad, upper + pad];
      const x = (v) =>
        margin.left +
        ((v - xDomain[0]) / span(xDomain)) *
          (width - margin.left - margin.right);
      const y = (v) =>
        height -
        margin.bottom -
        ((v - yDomain[0]) / span(yDomain)) *
          (height - margin.top - margin.bottom);
      const project = (axes) => [x(axes[0]), y(axes[two ? 1 : outputIndex])];
      svg.append(
        make("rect", {
          x: margin.left,
          y: margin.top,
          width: width - margin.left - margin.right,
          height: height - margin.top - margin.bottom,
          class: "chart-frame",
        }),
      );
      const ticks = width < 400 ? 3 : 5;
      for (let i = 0; i < ticks; i++) {
        const t = i / (ticks - 1),
          xv = xDomain[0] + t * span(xDomain),
          yv = yDomain[0] + t * span(yDomain);
        svg.append(
          make("line", {
            x1: x(xv),
            x2: x(xv),
            y1: margin.top,
            y2: height - margin.bottom,
            class: "chart-grid",
          }),
        );
        svg.append(
          make("line", {
            x1: margin.left,
            x2: width - margin.right,
            y1: y(yv),
            y2: y(yv),
            class: "chart-grid",
          }),
        );
        svg.append(
          make(
            "text",
            {
              x: x(xv),
              y: height - margin.bottom + 21,
              "text-anchor":
                i === 0 ? "start" : i === ticks - 1 ? "end" : "middle",
            },
            tick(xv),
          ),
        );
        svg.append(
          make(
            "text",
            { x: margin.left - 8, y: y(yv) + 4, "text-anchor": "end" },
            tick(yv),
          ),
        );
      }
      if (two) faces(project);
      else {
        let previous = -2;
        const path = points
          .map((point) => {
            const command = point.index === previous + 1 ? "L" : "M";
            previous = point.index;
            const p = project(point.axes);
            return `${command}${p[0]},${p[1]}`;
          })
          .join(" ");
        svg.append(make("path", { d: path, class: "response-line" }));
      }
      dots(project);
      svg.append(
        make(
          "text",
          { x: margin.left, y: height - 12 },
          label(fields.get(data.controls[0])),
        ),
      );
      svg.append(
        make(
          "text",
          { x: margin.left, y: 14 },
          label(fields.get(two ? data.controls[1] : data.output)),
        ),
      );
    }
    function surface() {
      svg.replaceChildren();
      const angle = (camera.rotation * Math.PI) / 180,
        tilt = (camera.tilt * Math.PI) / 180;
      const domains = [
        ...data.windows,
        upper === lower ? [lower - 1, upper + 1] : [lower, upper],
      ];
      const rotate = (axes) => {
        const [x, y, z] = axes.map(
          (v, i) => (v - domains[i][0]) / span(domains[i]),
        );
        const u = (x - 0.5) * Math.cos(angle) - (y - 0.5) * Math.sin(angle);
        const depth = (x - 0.5) * Math.sin(angle) + (y - 0.5) * Math.cos(angle);
        return [
          u,
          depth * Math.sin(tilt) - z * 0.85 * Math.cos(tilt),
          depth * Math.cos(tilt) + z * Math.sin(tilt),
        ];
      };
      const ground = domains[0].flatMap((x) =>
        domains[1].map((y) => [x, y, domains[2][0]]),
      );
      const front = ground.reduce((a, b) =>
        rotate(a)[2] > rotate(b)[2] ? a : b,
      );
      const left = ground.reduce((a, b) =>
        rotate(a)[0] < rotate(b)[0] ? a : b,
      );
      const starts = [front, front, left];
      const ends = domains.map((domain, axis) => {
        const end = [...starts[axis]];
        end[axis] =
          axis === 2 || end[axis] === domain[0] ? domain[1] : domain[0];
        return end;
      });
      const box = [
        ...points.map((point) => point.axes),
        ...starts,
        ...ends,
      ].map(rotate);
      const minimum = [0, 1].map((i) => Math.min(...box.map((p) => p[i])));
      const maximum = [0, 1].map((i) => Math.max(...box.map((p) => p[i])));
      const scale = Math.min(
        (width - 95) / (maximum[0] - minimum[0]),
        (height - 75) / (maximum[1] - minimum[1]),
      );
      const project = (axes) => {
        const p = rotate(axes);
        return [
          width / 2 + (p[0] - (minimum[0] + maximum[0]) / 2) * scale,
          height / 2 - 5 + (p[1] - (minimum[1] + maximum[1]) / 2) * scale,
          p[2],
        ];
      };
      // Axes retain the chosen display windows; the ceiling uses only checked cells.
      const labels = [];
      for (let axis = 0; axis < 3; axis++) {
        const a = project(starts[axis]),
          b = project(ends[axis]);
        const side = axis === 2 ? -1 : Math.sign(b[0] - a[0]) || 1;
        const field = fields.get([...data.controls, data.output][axis]);
        const symbol =
          field.symbol.length <= 18 ? field.symbol : ["x", "y", "output"][axis];
        svg.append(
          make("line", {
            x1: a[0],
            y1: a[1],
            x2: b[0],
            y2: b[1],
            class: "surface-axis",
          }),
        );
        for (const t of [0, 0.5, 1]) {
          const at = [...starts[axis]];
          at[axis] = domains[axis][0] + t * span(domains[axis]);
          const p = project(at);
          labels.push(
            make(
              "text",
              {
                x: p[0] + side * 8,
                y: p[1] + (axis === 2 ? 3 : 18),
                "text-anchor": side < 0 ? "end" : "start",
                class: "surface-axis-label",
              },
              tick(at[axis]),
            ),
          );
        }
        labels.push(
          make(
            "text",
            {
              x: b[0] + (axis === 2 ? 8 : 0),
              y: b[1] + (axis === 2 ? -8 : 34),
              "text-anchor": axis === 2 ? "start" : "middle",
              class: "surface-axis-label",
            },
            symbol,
          ),
        );
      }
      faces(project, true);
      dots(project);
      svg.append(...labels);
    }
    if (two && view === "surface") {
      const rotation = node("div", "surface-rotation");
      for (const [key, title, min, max] of [
        ["rotation", "Rotate surface", -180, 180],
        ["tilt", "Surface tilt", 10, 75],
      ]) {
        const field = node("label", "range-label", title);
        const input = node("input");
        input.type = "range";
        input.min = min;
        input.max = max;
        input.value = camera[key];
        input.setAttribute("aria-label", title);
        input.addEventListener("input", () => {
          camera[key] = Number(input.value);
          surface();
        });
        field.append(input);
        rotation.append(field);
      }
      container.append(rotation);
      surface();
    } else flat();
    container.append(svg);
    if (two) {
      container.append(
        node(
          "p",
          "chart-axis-key",
          `${label(fields.get(data.controls[0]))} × ${label(fields.get(data.controls[1]))} → ${label(fields.get(data.output))}`,
        ),
      );
      const legend = node("div", "chart-colour-key");
      const gradient = node("span", "chart-gradient");
      gradient.style.background = `linear-gradient(to right, ${colour(0)}, ${colour(1)})`;
      legend.append(
        node("span", "", tick(lower)),
        gradient,
        node("span", "", tick(upper)),
        node("span", "", label(fields.get(data.output))),
      );
      container.append(
        legend,
        node(
          "p",
          "input-help",
          `${view === "surface" ? "Surface faces" : "Shaded cells"} interpolate checked corners. Boundaries are approximate; missing cells stay open.`,
        ),
      );
    }
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
      const option = node("option", "", text(point));
      option.value = index;
      sampleSelect.append(option);
    });
    sampleSelect.addEventListener("change", () => {
      if (sampleSelect.value !== "")
        onSelect(points[Number(sampleSelect.value)].point);
    });
    container.append(sampleSelect, readout);
  }
  return { render };
})();
