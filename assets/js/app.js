(function () {
  const state = {
    datasetName: "",
    headers: [],
    rows: [],
    profile: null,
    problemType: null,
    problemRationale: "",
    processor: null,
    selectedAlgorithm: null,
    trainedModel: null,
    split: null,
    evaluation: null,
  };

  const dom = {};

  document.addEventListener("DOMContentLoaded", () => {
    bindElements();
    bindEvents();
  });

  function bindElements() {
    [
      "datasetInput",
      "datasetStatus",
      "targetSelect",
      "analyzeBtn",
      "emptyState",
      "datasetSection",
      "algorithmSection",
      "resultSection",
      "predictSection",
      "problemBadge",
      "healthCards",
      "missingChart",
      "targetChart",
      "featureChart",
      "distributionChart",
      "missingCaption",
      "targetCaption",
      "featureCaption",
      "distributionCaption",
      "preprocessTimeline",
      "previewTable",
      "algorithmCards",
      "algorithmExplanation",
      "trainBtn",
      "clusterControl",
      "clusterCount",
      "modelBadge",
      "metricCards",
      "evaluationChart",
      "evaluationChartTitle",
      "evaluationCaption",
      "importanceList",
      "trainingNarrative",
      "predictForm",
      "predictBtn",
      "fillExampleBtn",
      "predictionOutput",
    ].forEach((id) => {
      dom[id] = document.getElementById(id);
    });
  }

  function bindEvents() {
    dom.datasetInput.addEventListener("change", async (event) => {
      const file = event.target.files?.[0];
      if (!file) return;
      const text = await file.text();
      loadDataset(text, file.name);
    });

    document.querySelectorAll("[data-sample]").forEach((button) => {
      button.addEventListener("click", () => {
        const sample = window.SampleDatasets[button.dataset.sample];
        loadDataset(sample.csv, sample.name);
      });
    });

    dom.analyzeBtn.addEventListener("click", analyzeDataset);
    dom.trainBtn.addEventListener("click", trainSelectedAlgorithm);
    dom.fillExampleBtn.addEventListener("click", fillExampleRow);
    dom.predictBtn.addEventListener("click", predictOneRow);
  }

  function loadDataset(text, name) {
    try {
      const parsed = window.MLData.parseCSV(text);
      state.datasetName = name;
      state.headers = parsed.headers;
      state.rows = parsed.rows;
      state.profile = window.MLData.profileDataset(state.rows, state.headers);
      state.problemType = null;
      state.processor = null;
      state.selectedAlgorithm = null;
      state.trainedModel = null;

      dom.datasetStatus.textContent = `${name} loaded`;
      dom.targetSelect.disabled = false;
      dom.analyzeBtn.disabled = false;
      renderTargetOptions();
      hideAfterUploadReset();
    } catch (error) {
      alert(error.message);
    }
  }

  function renderTargetOptions() {
    const options = [`<option value="">No target / clustering</option>`]
      .concat(state.headers.map((header) => `<option value="${escapeAttribute(header)}">${escapeHtml(header)}</option>`))
      .join("");
    dom.targetSelect.innerHTML = options;
    if (state.headers.length) dom.targetSelect.value = state.headers[state.headers.length - 1];
  }

  function hideAfterUploadReset() {
    dom.emptyState.classList.remove("hidden");
    dom.datasetSection.classList.add("hidden");
    dom.algorithmSection.classList.add("hidden");
    dom.resultSection.classList.add("hidden");
    dom.predictSection.classList.add("hidden");
    dom.trainBtn.disabled = true;
    dom.predictionOutput.innerHTML = `<strong>Prediction will appear here.</strong>`;
  }

  function analyzeDataset() {
    const targetColumn = dom.targetSelect.value;
    const requestedMode = document.querySelector("input[name='learningMode']:checked").value;
    const mode = requestedMode === "supervised" && !targetColumn ? "auto" : requestedMode;
    const problem = window.MLData.inferProblemType(state.rows, targetColumn, mode);

    if (requestedMode === "supervised" && !targetColumn) {
      alert("Choose a target column for supervised learning.");
      return;
    }

    state.problemType = problem.problemType;
    state.problemRationale = problem.rationale;
    state.processor = window.MLData.buildPreprocessor(state.rows, state.headers, targetColumn, state.problemType);
    state.selectedAlgorithm = null;
    state.trainedModel = null;
    state.split = null;
    state.evaluation = null;

    renderDatasetSection();
    renderAlgorithmSection();
    dom.emptyState.classList.add("hidden");
    dom.datasetSection.classList.remove("hidden");
    dom.algorithmSection.classList.remove("hidden");
    dom.resultSection.classList.add("hidden");
    dom.predictSection.classList.add("hidden");
  }

  function renderDatasetSection() {
    const processor = state.processor;
    const profile = state.profile;
    const targetColumn = processor.targetColumn;
    dom.problemBadge.textContent = `${capitalize(state.problemType)} detected`;

    dom.healthCards.innerHTML = [
      metricCard("Rows", profile.rowCount, "Raw examples in the uploaded file"),
      metricCard("Columns", profile.columnCount, `${profile.numericCount} numeric, ${profile.categoricalCount} categorical`),
      metricCard("Features", processor.featureColumns.length, `${processor.featureNames.length} model-ready columns after encoding`),
      metricCard("Task", capitalize(state.problemType), state.problemRationale),
    ].join("");

    const missingEntries = state.headers
      .map((header) => ({ label: header, value: profile.types[header]?.missing || 0 }))
      .sort((a, b) => b.value - a.value)
      .slice(0, 8);
    window.MLCharts.drawBarChart(
      dom.missingChart,
      missingEntries.map((entry) => entry.label),
      missingEntries.map((entry) => entry.value),
      { emptyMessage: "No missing values" }
    );
    dom.missingCaption.textContent = `${profile.missingTotal} total missing`;

    const target = window.MLData.targetSummary(processor.rows, targetColumn, state.problemType);
    if (state.problemType === "regression") {
      window.MLCharts.drawHistogram(dom.targetChart, target.numericValues, { emptyMessage: "No target data" });
    } else if (state.problemType === "classification") {
      window.MLCharts.drawBarChart(dom.targetChart, target.labels, target.values, { emptyMessage: "No target classes" });
    } else {
      window.MLCharts.drawEmpty(dom.targetChart, "Clustering has no target");
    }
    dom.targetCaption.textContent = target.caption;

    window.MLCharts.drawBarChart(dom.featureChart, ["Numeric", "Categorical", "Encoded"], [
      processor.numericColumns.length,
      processor.categoricalColumns.length,
      processor.featureNames.length,
    ]);
    dom.featureCaption.textContent = `${processor.featureColumns.length} original feature columns`;

    const numericColumn = processor.numericColumns[0];
    if (numericColumn) {
      const values = processor.rows.map((row) => window.MLData.toNumber(row[numericColumn])).filter(Number.isFinite);
      window.MLCharts.drawHistogram(dom.distributionChart, values);
      dom.distributionCaption.textContent = numericColumn;
    } else {
      window.MLCharts.drawEmpty(dom.distributionChart, "No numeric feature columns");
      dom.distributionCaption.textContent = "Categorical-only dataset";
    }

    dom.preprocessTimeline.innerHTML = window.MLData
      .preprocessingSummary(processor)
      .map(
        (item, index) => `<div class="timeline-item">
          <span class="timeline-index">${index + 1}</span>
          <div><strong>${escapeHtml(item.title)}</strong><p>${escapeHtml(item.body)}</p></div>
        </div>`
      )
      .join("");

    window.MLCharts.renderTable(dom.previewTable, state.headers, profile.preview);
  }

  function renderAlgorithmSection() {
    const algorithms = window.MLExplanations.getAlgorithms(state.problemType);
    dom.clusterControl.classList.toggle("hidden", state.problemType !== "unsupervised");
    dom.algorithmCards.innerHTML = algorithms
      .map(
        (algorithm) => `<button type="button" class="algorithm-card" data-algorithm="${algorithm.id}">
          <small>${escapeHtml(algorithm.family)}</small>
          <strong>${escapeHtml(algorithm.name)}</strong>
          <span>${escapeHtml(algorithm.short)}</span>
        </button>`
      )
      .join("");

    dom.algorithmCards.querySelectorAll("[data-algorithm]").forEach((button) => {
      button.addEventListener("click", () => {
        state.selectedAlgorithm = button.dataset.algorithm;
        dom.algorithmCards.querySelectorAll(".algorithm-card").forEach((card) => card.classList.remove("is-selected"));
        button.classList.add("is-selected");
        renderAlgorithmExplanation();
        dom.trainBtn.disabled = false;
      });
    });

    dom.algorithmExplanation.innerHTML = `
      <p class="eyebrow">Algorithm explanation</p>
      <h3>Select an algorithm to see the concept before training.</h3>
    `;
    dom.trainBtn.disabled = true;
  }

  function renderAlgorithmExplanation() {
    const algorithm = window.MLExplanations.getAlgorithm(state.problemType, state.selectedAlgorithm);
    if (!algorithm) return;
    dom.algorithmExplanation.innerHTML = `
      <p class="eyebrow">${escapeHtml(algorithm.family)}</p>
      <h3>${escapeHtml(algorithm.name)}</h3>
      <p>${escapeHtml(algorithm.concept)}</p>
      <div class="explanation-grid">
        <div class="explanation-box">
          <strong>Advantages</strong>
          <ul>${algorithm.advantages.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
        </div>
        <div class="explanation-box">
          <strong>Process steps</strong>
          <ol>${algorithm.steps.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ol>
        </div>
        <div class="explanation-box">
          <strong>In this app</strong>
          <ul>
            <li>The data is cleaned before training.</li>
            <li>The selected model is evaluated on holdout rows.</li>
            <li>The prediction section explains the model's choice.</li>
          </ul>
        </div>
      </div>
    `;
  }

  function trainSelectedAlgorithm() {
    if (!state.processor || !state.selectedAlgorithm) return;
    try {
      const processor = state.processor;
      const algorithm = window.MLExplanations.getAlgorithm(state.problemType, state.selectedAlgorithm);
      const capped = capTrainingRows(processor.X, processor.y, processor.rows, state.selectedAlgorithm === "dbscan" ? 650 : 2500);
      let model;

      if (state.problemType === "unsupervised") {
        const k = clamp(Number(dom.clusterCount.value) || 3, 2, 8);
        model = window.MLModels.train("unsupervised", state.selectedAlgorithm, capped.X, null, { k });
        state.split = { trainX: capped.X, trainRows: capped.rows };
        state.evaluation = summarizeClusters(model.assignments);
      } else {
        const split = window.MLModels.splitDataset(capped.X, capped.y, 0.25, 17);
        model = window.MLModels.train(state.problemType, state.selectedAlgorithm, split.trainX, split.trainY, {
          classCount: processor.targetEncoder?.classes.length,
        });
        state.split = split;
        state.evaluation = window.MLModels.evaluate(
          state.problemType,
          model,
          split.testX,
          split.testY,
          processor.targetEncoder?.classes || []
        );
        state.split.testRows = split.testIndices.map((index) => capped.rows[index]);
      }

      state.trainedModel = model;
      renderResults(algorithm, capped.X.length);
      renderPredictionSection();
      dom.resultSection.classList.remove("hidden");
      dom.predictSection.classList.remove("hidden");
    } catch (error) {
      alert(error.message);
    }
  }

  function renderResults(algorithm, rowCount) {
    dom.modelBadge.textContent = algorithm.name;
    const model = state.trainedModel;
    const evaluation = state.evaluation;

    if (state.problemType === "regression") {
      dom.metricCards.innerHTML = [
        metricCard("MAE", window.MLCharts.formatNumber(evaluation.mae), "Average absolute prediction error"),
        metricCard("RMSE", window.MLCharts.formatNumber(evaluation.rmse), "Punishes larger errors more strongly"),
        metricCard("R2", window.MLCharts.formatNumber(evaluation.r2), "How much target variation the model explains"),
        metricCard("Test rows", evaluation.predictions.length, "Holdout examples used for evaluation"),
      ].join("");
      dom.evaluationChartTitle.textContent = "Actual vs predicted";
      dom.evaluationCaption.textContent = "Holdout rows";
      const points = evaluation.predictions.map((prediction, index) => [state.split.testY[index], prediction]);
      window.MLCharts.drawScatter(dom.evaluationChart, points, [], { xLabel: "Actual", yLabel: "Predicted" });
    } else if (state.problemType === "classification") {
      dom.metricCards.innerHTML = [
        metricCard("Accuracy", `${Math.round(evaluation.accuracy * 100)}%`, "Correct predictions on holdout rows"),
        metricCard("Macro F1", `${Math.round(evaluation.macroF1 * 100)}%`, "Balances precision and recall across classes"),
        metricCard("Classes", state.processor.targetEncoder.classes.length, "Target labels learned by the model"),
        metricCard("Test rows", evaluation.predictions.length, "Holdout examples used for evaluation"),
      ].join("");
      dom.evaluationChartTitle.textContent = "Confusion view";
      dom.evaluationCaption.textContent = "Actual class vs predicted class";
      const labels = state.processor.targetEncoder.classes;
      const chartLabels = [];
      const chartValues = [];
      evaluation.matrix.forEach((row, actualIndex) => {
        row.forEach((value, predictedIndex) => {
          chartLabels.push(`${labels[actualIndex]} -> ${labels[predictedIndex]}`);
          chartValues.push(value);
        });
      });
      window.MLCharts.drawBarChart(dom.evaluationChart, chartLabels, chartValues, { emptyMessage: "No confusion data" });
    } else {
      const clusterLabels = state.trainedModel.assignments;
      dom.metricCards.innerHTML = [
        metricCard("Clusters", evaluation.clusterCount, "Groups discovered without a target column"),
        metricCard("Noise", evaluation.noiseCount, "Rows DBSCAN treated as isolated points"),
        metricCard("Rows", rowCount, "Rows used for clustering"),
        metricCard("Features", state.processor.featureNames.length, "Model-ready feature dimensions"),
      ].join("");
      dom.evaluationChartTitle.textContent = "Cluster map";
      dom.evaluationCaption.textContent = "PCA projection of model-ready features";
      window.MLCharts.drawScatter(dom.evaluationChart, window.MLModels.pca2D(state.split.trainX), clusterLabels, {
        xLabel: "PCA 1",
        yLabel: "PCA 2",
      });
    }

    window.MLCharts.renderImportance(dom.importanceList, model.importances, state.processor.featureNames);
    dom.trainingNarrative.innerHTML = buildTrainingNarrative(algorithm, rowCount);
  }

  function buildTrainingNarrative(algorithm, rowCount) {
    const processor = state.processor;
    const taskLine =
      state.problemType === "unsupervised"
        ? `The app ignored the target selector and searched for structure across ${processor.featureColumns.length} feature column(s).`
        : `The app used "${escapeHtml(processor.targetColumn)}" as the target and held out 25% of rows for testing.`;
    return `
      <p><strong>${escapeHtml(algorithm.name)}</strong> trained on ${rowCount} prepared row(s). ${taskLine}</p>
      <p>${escapeHtml(state.problemRationale)}</p>
      <p>The preprocessing stage converted the raw dataset into ${processor.featureNames.length} numeric model-ready feature(s). That is why text columns can be used by mathematical algorithms.</p>
    `;
  }

  function renderPredictionSection() {
    const processor = state.processor;
    dom.predictForm.innerHTML = processor.featureColumns
      .map((column) => {
        const safeId = `input_${column.replace(/[^a-z0-9]/gi, "_")}`;
        if (processor.categoricalColumns.includes(column)) {
          const options = processor.categoricalStats[column].categories
            .filter((category) => category !== "__other__")
            .map((category) => `<option value="${escapeAttribute(category)}">${escapeHtml(category)}</option>`)
            .join("");
          return `<div class="predict-field">
            <label for="${safeId}">${escapeHtml(column)}</label>
            <select id="${safeId}" data-feature="${escapeAttribute(column)}">${options}</select>
          </div>`;
        }
        const median = processor.numericStats[column]?.median ?? "";
        return `<div class="predict-field">
          <label for="${safeId}">${escapeHtml(column)}</label>
          <input id="${safeId}" type="number" step="any" data-feature="${escapeAttribute(column)}" value="${escapeAttribute(median)}" />
        </div>`;
      })
      .join("");
    fillExampleRow();
  }

  function fillExampleRow() {
    if (!state.processor) return;
    const example =
      state.problemType === "unsupervised"
        ? state.processor.rows[0]
        : state.split?.testRows?.[0] || state.processor.rows[0];
    if (!example) return;
    dom.predictForm.querySelectorAll("[data-feature]").forEach((input) => {
      const feature = input.dataset.feature;
      input.value = example[feature] ?? input.value;
    });
  }

  function predictOneRow() {
    if (!state.trainedModel || !state.processor) return;
    const raw = {};
    dom.predictForm.querySelectorAll("[data-feature]").forEach((input) => {
      raw[input.dataset.feature] = input.value;
    });
    const row = state.processor.transformRow(raw);
    const rawPrediction = state.trainedModel.predictOne(row);
    let result = rawPrediction;
    let classIndex = rawPrediction;
    if (state.problemType === "classification") {
      result = state.processor.targetEncoder.classes[rawPrediction];
    }
    const explanation = window.MLModels.explainPrediction(
      state.problemType,
      state.trainedModel,
      row,
      state.processor,
      state.problemType === "classification" ? { classIndex } : rawPrediction,
      state.processor.targetEncoder?.classes || []
    );
    dom.predictionOutput.innerHTML = `
      <strong>${escapeHtml(explanation.headline)}</strong>
      <span>${escapeHtml(explanation.why)}</span>
      <span>${escapeHtml(explanation.whyNot)}</span>
      ${
        explanation.signals.length
          ? `<div><b>Main signals:</b><ul>${explanation.signals
              .map((signal) => `<li>${escapeHtml(signal.name)} was ${escapeHtml(signal.direction)} for this prediction.</li>`)
              .join("")}</ul></div>`
          : ""
      }
      ${state.problemType === "classification" ? `<span>Final label returned to the user: <b>${escapeHtml(result)}</b></span>` : ""}
    `;
  }

  function capTrainingRows(X, y, rows, limit) {
    if (X.length <= limit) return { X, y, rows };
    const step = X.length / limit;
    const indices = Array.from({ length: limit }, (_, index) => Math.floor(index * step));
    return {
      X: indices.map((index) => X[index]),
      y: y ? indices.map((index) => y[index]) : null,
      rows: indices.map((index) => rows[index]),
    };
  }

  function summarizeClusters(assignments) {
    const unique = Array.from(new Set(assignments.filter((label) => label !== -1)));
    return {
      clusterCount: unique.length,
      noiseCount: assignments.filter((label) => label === -1).length,
    };
  }

  function metricCard(label, value, note) {
    return `<article class="metric-card">
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(String(value))}</strong>
      <p>${escapeHtml(note)}</p>
    </article>`;
  }

  function capitalize(value) {
    return String(value || "").charAt(0).toUpperCase() + String(value || "").slice(1);
  }

  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
  }

  function escapeHtml(value) {
    return window.MLCharts.escapeHtml(value);
  }

  function escapeAttribute(value) {
    return escapeHtml(value).replace(/"/g, "&quot;");
  }
})();
