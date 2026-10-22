"use strict";
// Geometry owns ranges, adjacency, and projections; it does not create UI elements.
const chartGeometry = (() => {
  class PlotRange {
    constructor([lower, upper]) {
      this.lower = lower;
      this.upper = upper;
    }
    static enclosing(values) {
      return new PlotRange([Math.min(...values), Math.max(...values)]);
    }
    get endpoints() {
      return [this.lower, this.upper];
    }
    get span() {
      return this.upper - this.lower;
    }
    get midpoint() {
      return (this.lower + this.upper) / 2;
    }
    at(fraction) {
      return this.lower + fraction * this.span;
    }
    fraction(value) {
      return this.span === 0 ? 0.5 : (value - this.lower) / this.span;
    }
    padded() {
      const padding =
        this.span === 0
          ? Math.max(Math.abs(this.lower) * 0.05, 1)
          : this.span * 0.08;
      return new PlotRange([this.lower - padding, this.upper + padding]);
    }
    nonzero() {
      return this.span === 0
        ? new PlotRange([this.lower - 1, this.upper + 1])
        : this;
    }
  }

  class SampleGrid {
    constructor(data) {
      this.points = data.points;
      this.controls = data.controls;
      this.output = data.output;
      this.size = data.grid_size;
      this.windows = data.windows.map((bounds) => new PlotRange(bounds));
      this.outputRange = PlotRange.enclosing(
        this.points.map((point) => this.value(point)),
      );
      this.byIndex = new Map(this.points.map((point) => [point.index, point]));
    }
    get joint() {
      return this.controls.length === 2;
    }
    get paths() {
      return [...this.controls, this.output];
    }
    value(point) {
      return point.axes[this.controls.length];
    }
    *cells() {
      if (!this.joint) return;
      for (let row = 0; row < this.size - 1; row++) {
        for (let column = 0; column < this.size - 1; column++) {
          const index = row * this.size + column;
          const corners = [
            index,
            index + this.size,
            index + this.size + 1,
            index + 1,
          ].map((index) => this.byIndex.get(index));
          // A missing or rejected corner leaves an open cell.
          if (corners.every(Boolean)) yield corners;
        }
      }
    }
    *segments() {
      let segment = [],
        previous = -2;
      for (const point of this.points) {
        if (point.index !== previous + 1 && segment.length) {
          yield segment;
          segment = [];
        }
        segment.push(point);
        previous = point.index;
      }
      if (segment.length) yield segment;
    }
    mesh(projection) {
      const cells = [...this.cells()].map((corners) => {
        const vertices = corners.map((point) => projection.project(point.axes));
        return {
          vertices,
          level:
            corners.reduce(
              (sum, point) =>
                sum + this.outputRange.fraction(this.value(point)),
              0,
            ) / 4,
          depth: projection.depthOrdered
            ? vertices.reduce((sum, point) => sum + point[2], 0)
            : 0,
        };
      });
      if (projection.depthOrdered) cells.sort((a, b) => a.depth - b.depth);
      return cells;
    }
  }

  class FlatProjection {
    depthOrdered = false;
    constructor(grid, width, height) {
      this.grid = grid;
      this.width = width;
      this.height = height;
      this.margin = { left: 58, right: 22, top: 25, bottom: 60 };
      this.horizontal = grid.windows[0];
      this.vertical = grid.joint ? grid.windows[1] : grid.outputRange.padded();
    }
    get frame() {
      return {
        x: this.margin.left,
        y: this.margin.top,
        width: this.width - this.margin.left - this.margin.right,
        height: this.height - this.margin.top - this.margin.bottom,
      };
    }
    x(value) {
      return (
        this.margin.left + this.horizontal.fraction(value) * this.frame.width
      );
    }
    y(value) {
      return (
        this.height -
        this.margin.bottom -
        this.vertical.fraction(value) * this.frame.height
      );
    }
    project(axes) {
      return [
        this.x(axes[0]),
        this.y(axes[this.grid.joint ? 1 : this.grid.controls.length]),
      ];
    }
  }

  class SurfaceAxis {
    constructor(range, index, start) {
      this.range = range;
      this.index = index;
      this.start = start;
      this.end = [...start];
      this.end[index] =
        index === 2 || start[index] === range.lower ? range.upper : range.lower;
    }
    at(fraction) {
      const point = [...this.start];
      point[this.index] = this.range.at(fraction);
      return point;
    }
  }

  class SurfaceProjection {
    depthOrdered = true;
    constructor(grid, camera, width, height) {
      this.angle = (camera.rotation * Math.PI) / 180;
      this.tilt = (camera.tilt * Math.PI) / 180;
      this.width = width;
      this.height = height;
      this.ranges = [...grid.windows, grid.outputRange.nonzero()];
      const ground = this.ranges[0].endpoints.flatMap((x) =>
        this.ranges[1].endpoints.map((y) => [x, y, this.ranges[2].lower]),
      );
      const front = ground.reduce((a, b) =>
        this.rotate(a)[2] > this.rotate(b)[2] ? a : b,
      );
      const left = ground.reduce((a, b) =>
        this.rotate(a)[0] < this.rotate(b)[0] ? a : b,
      );
      this.axes = this.ranges.map(
        (range, index) =>
          new SurfaceAxis(range, index, index === 2 ? left : front),
      );
      const box = [
        ...grid.points.map((point) => point.axes),
        ...this.axes.flatMap((axis) => [axis.start, axis.end]),
      ].map((point) => this.rotate(point));
      this.horizontal = PlotRange.enclosing(box.map((point) => point[0]));
      this.vertical = PlotRange.enclosing(box.map((point) => point[1]));
      this.scale = Math.min(
        (width - 95) / this.horizontal.span,
        (height - 75) / this.vertical.span,
      );
    }
    rotate(axes) {
      const [x, y, z] = axes.map((value, axis) =>
        this.ranges[axis].fraction(value),
      );
      const horizontal =
        (x - 0.5) * Math.cos(this.angle) - (y - 0.5) * Math.sin(this.angle);
      const depth =
        (x - 0.5) * Math.sin(this.angle) + (y - 0.5) * Math.cos(this.angle);
      return [
        horizontal,
        depth * Math.sin(this.tilt) - z * 0.85 * Math.cos(this.tilt),
        depth * Math.cos(this.tilt) + z * Math.sin(this.tilt),
      ];
    }
    project(axes) {
      const [x, y, depth] = this.rotate(axes);
      return [
        this.width / 2 + (x - this.horizontal.midpoint) * this.scale,
        this.height / 2 - 5 + (y - this.vertical.midpoint) * this.scale,
        depth,
      ];
    }
  }
  return { SampleGrid, FlatProjection, SurfaceProjection };
})();
