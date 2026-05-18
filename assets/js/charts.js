(function () {
  const COLORS = ["#0f9b8e", "#e9654b", "#4f5bd5", "#d89b1d", "#334155", "#7c3aed", "#0f766e", "#c2410c"];

  function prepareCanvas(canvas) {
    const ratio = window.devicePixelRatio || 1;
    const width = canvas.clientWidth || canvas.parentElement.clientWidth || 420;
    const height = Number(canvas.getAttribute("height")) || 220;
    canvas.width = width * ratio;
    canvas.height = height * ratio;
    const context = canvas.getContext("2d");
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, width, height);
    context.font = "12px Inter, system-ui, sans-serif";
    context.lineWidth = 1;
    return { context, width, height };
  }

  function drawEmpty(canvas, message) {
    const { context, width, height } = prepareCanvas(canvas);
    context.fillStyle = "#647084";
    context.textAlign = "center";
    context.fillText(message || "No chart data", width / 2, height / 2);
  }

  function drawBarChart(canvas, labels, values, options = {}) {
    if (!labels?.length || !values?.length) return drawEmpty(canvas, options.emptyMessage);
    const { context, width, height } = prepareCanvas(canvas);
    const padding = { top: 18, right: 18, bottom: 46, left: 44 };
    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;
    const maxValue = Math.max(...values, 1);
    const barWidth = Math.max(14, chartWidth / values.length - 12);

    drawAxes(context, padding, width, height);

    values.forEach((value, index) => {
      const x = padding.left + index * (chartWidth / values.length) + 6;
      const barHeight = (value / maxValue) * chartHeight;
      const y = padding.top + chartHeight - barHeight;
      context.fillStyle = options.colors?.[index] || COLORS[index % COLORS.length];
      context.fillRect(x, y, barWidth, barHeight);
      context.fillStyle = "#1d2433";
      context.textAlign = "center";
      context.fillText(formatNumber(value), x + barWidth / 2, y - 6);
      context.save();
      context.translate(x + barWidth / 2, height - 16);
      context.rotate(labels[index].length > 10 ? -0.38 : 0);
      context.fillStyle = "#647084";
      context.fillText(shorten(labels[index], 16), 0, 0);
      context.restore();
    });
  }

  function drawHistogram(canvas, values, options = {}) {
    const numeric = values?.filter(Number.isFinite) || [];
    if (!numeric.length) return drawEmpty(canvas, options.emptyMessage || "No numeric values");
    const min = Math.min(...numeric);
    const max = Math.max(...numeric);
    if (min === max) return drawBarChart(canvas, [String(min)], [numeric.length], options);
    const bins = Math.min(options.bins || 8, Math.max(4, Math.ceil(Math.sqrt(numeric.length))));
    const counts = Array.from({ length: bins }, () => 0);
    numeric.forEach((value) => {
      const index = Math.min(bins - 1, Math.floor(((value - min) / (max - min)) * bins));
      counts[index] += 1;
    });
    const labels = counts.map((_, index) => {
      const start = min + ((max - min) / bins) * index;
      const end = min + ((max - min) / bins) * (index + 1);
      return `${formatNumber(start)}-${formatNumber(end)}`;
    });
    drawBarChart(canvas, labels, counts, options);
  }

  function drawScatter(canvas, points, labels = [], options = {}) {
    if (!points?.length) return drawEmpty(canvas, options.emptyMessage || "No points");
    const { context, width, height } = prepareCanvas(canvas);
    const padding = { top: 18, right: 18, bottom: 36, left: 44 };
    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;
    const xs = points.map((point) => point[0]);
    const ys = points.map((point) => point[1]);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);

    drawAxes(context, padding, width, height);

    points.forEach((point, index) => {
      const x = padding.left + normalize(point[0], minX, maxX) * chartWidth;
      const y = padding.top + chartHeight - normalize(point[1], minY, maxY) * chartHeight;
      const label = labels[index] ?? 0;
      context.fillStyle = label === -1 ? "#647084" : COLORS[Math.abs(label) % COLORS.length];
      context.beginPath();
      context.arc(x, y, 4.5, 0, Math.PI * 2);
      context.fill();
    });

    context.fillStyle = "#647084";
    context.textAlign = "left";
    context.fillText(options.xLabel || "Component 1", padding.left, height - 10);
    context.save();
    context.translate(14, padding.top + 78);
    context.rotate(-Math.PI / 2);
    context.fillText(options.yLabel || "Component 2", 0, 0);
    context.restore();
  }

  function renderTable(container, headers, rows) {
    if (!rows.length) {
      container.innerHTML = `<p class="notice">No rows to display.</p>`;
      return;
    }
    const head = headers.map((header) => `<th>${escapeHtml(header)}</th>`).join("");
    const body = rows
      .map(
        (row) =>
          `<tr>${headers
            .map((header) => `<td>${escapeHtml(row[header] === undefined ? "" : String(row[header]))}</td>`)
            .join("")}</tr>`
      )
      .join("");
    container.innerHTML = `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
  }

  function renderImportance(container, importances, featureNames) {
    const entries = featureNames
      .map((name, index) => ({ name, value: Math.abs(importances?.[index] || 0) }))
      .sort((a, b) => b.value - a.value)
      .slice(0, 8);
    const max = Math.max(...entries.map((entry) => entry.value), 1e-9);
    if (!entries.length || entries.every((entry) => entry.value === 0)) {
      container.innerHTML = `<p class="notice">This model did not expose a strong feature-importance signal.</p>`;
      return;
    }
    container.innerHTML = entries
      .map(
        (entry) => `<div class="importance-row">
          <strong>${escapeHtml(shorten(entry.name, 28))}</strong>
          <span class="bar-track"><span class="bar-fill" style="width: ${(entry.value / max) * 100}%"></span></span>
          <span>${Math.round((entry.value / max) * 100)}%</span>
        </div>`
      )
      .join("");
  }

  function drawAxes(context, padding, width, height) {
    context.strokeStyle = "#dbe2ec";
    context.beginPath();
    context.moveTo(padding.left, padding.top);
    context.lineTo(padding.left, height - padding.bottom);
    context.lineTo(width - padding.right, height - padding.bottom);
    context.stroke();
  }

  function normalize(value, min, max) {
    if (max === min) return 0.5;
    return (value - min) / (max - min);
  }

  function formatNumber(value) {
    if (!Number.isFinite(value)) return "-";
    if (Math.abs(value) >= 1000) return value.toFixed(0);
    if (Math.abs(value) >= 10) return value.toFixed(1).replace(/\.0$/, "");
    return value.toFixed(2).replace(/\.?0+$/, "");
  }

  function shorten(value, length) {
    const text = String(value);
    return text.length > length ? `${text.slice(0, length - 1)}...` : text;
  }

  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, (char) => {
      const map = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" };
      return map[char];
    });
  }

  window.MLCharts = {
    drawBarChart,
    drawHistogram,
    drawScatter,
    drawEmpty,
    renderTable,
    renderImportance,
    formatNumber,
    escapeHtml,
    shorten,
    COLORS,
  };
})();
