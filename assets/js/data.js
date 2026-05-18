(function () {
  const MISSING_VALUES = new Set(["", "na", "n/a", "null", "none", "undefined", "?"]);

  function parseCSV(text) {
    const delimiter = detectDelimiter(text);
    const rows = [];
    let field = "";
    let record = [];
    let inQuotes = false;

    for (let index = 0; index < text.length; index += 1) {
      const char = text[index];
      const next = text[index + 1];

      if (char === '"') {
        if (inQuotes && next === '"') {
          field += '"';
          index += 1;
        } else {
          inQuotes = !inQuotes;
        }
      } else if (char === delimiter && !inQuotes) {
        record.push(field.trim());
        field = "";
      } else if ((char === "\n" || char === "\r") && !inQuotes) {
        if (char === "\r" && next === "\n") index += 1;
        record.push(field.trim());
        if (record.some((value) => value !== "")) rows.push(record);
        field = "";
        record = [];
      } else {
        field += char;
      }
    }

    record.push(field.trim());
    if (record.some((value) => value !== "")) rows.push(record);

    if (rows.length < 2) {
      throw new Error("The CSV needs one header row and at least one data row.");
    }

    const headers = makeUniqueHeaders(rows[0]);
    const objects = rows.slice(1).map((row) => {
      const entry = {};
      headers.forEach((header, columnIndex) => {
        entry[header] = row[columnIndex] ?? "";
      });
      return entry;
    });

    return { headers, rows: objects, delimiter };
  }

  function detectDelimiter(text) {
    const firstLine = text.split(/\r?\n/).find((line) => line.trim().length > 0) || "";
    const delimiters = [",", ";", "\t", "|"];
    return delimiters
      .map((delimiter) => ({ delimiter, count: firstLine.split(delimiter).length }))
      .sort((a, b) => b.count - a.count)[0].delimiter;
  }

  function makeUniqueHeaders(headers) {
    const seen = new Map();
    return headers.map((header, index) => {
      const base = header.trim() || `column_${index + 1}`;
      const count = seen.get(base) || 0;
      seen.set(base, count + 1);
      return count === 0 ? base : `${base}_${count + 1}`;
    });
  }

  function isMissing(value) {
    return MISSING_VALUES.has(String(value ?? "").trim().toLowerCase());
  }

  function toNumber(value) {
    if (isMissing(value)) return NaN;
    const cleaned = String(value).replace(/[$,%]/g, "").replace(/,/g, "").trim();
    if (cleaned === "") return NaN;
    const numeric = Number(cleaned);
    return Number.isFinite(numeric) ? numeric : NaN;
  }

  function inferColumnTypes(rows, headers) {
    const types = {};
    headers.forEach((header) => {
      const values = rows.map((row) => row[header]);
      const nonMissing = values.filter((value) => !isMissing(value));
      const numericValues = nonMissing.map(toNumber).filter(Number.isFinite);
      const numericRatio = nonMissing.length === 0 ? 0 : numericValues.length / nonMissing.length;
      const uniqueValues = new Set(nonMissing.map((value) => String(value).trim())).size;
      types[header] = {
        kind: numericRatio >= 0.85 ? "numeric" : "categorical",
        missing: values.length - nonMissing.length,
        unique: uniqueValues,
        numericRatio,
      };
    });
    return types;
  }

  function profileDataset(rows, headers) {
    const types = inferColumnTypes(rows, headers);
    const numericCount = headers.filter((header) => types[header].kind === "numeric").length;
    const categoricalCount = headers.length - numericCount;
    const missingTotal = headers.reduce((sum, header) => sum + types[header].missing, 0);

    return {
      rowCount: rows.length,
      columnCount: headers.length,
      numericCount,
      categoricalCount,
      missingTotal,
      types,
      preview: rows.slice(0, 8),
    };
  }

  function inferProblemType(rows, targetColumn, requestedMode) {
    if (requestedMode === "unsupervised" || !targetColumn) {
      return {
        problemType: "unsupervised",
        rationale: "No target is required, so the app will look for natural groups in the feature space.",
      };
    }

    const types = inferColumnTypes(rows, [targetColumn]);
    const targetType = types[targetColumn];
    const values = rows.map((row) => row[targetColumn]).filter((value) => !isMissing(value));

    if (targetType.kind === "categorical") {
      return {
        problemType: "classification",
        rationale: `The target "${targetColumn}" contains labels or categories, so the task is classification.`,
      };
    }

    const numbers = values.map(toNumber).filter(Number.isFinite);
    const uniqueNumbers = Array.from(new Set(numbers.map((value) => String(value))));
    const integerish = numbers.every((value) => Math.abs(value - Math.round(value)) < 1e-9);
    const smallClassLikeTarget =
      integerish && uniqueNumbers.length <= Math.max(2, Math.min(20, Math.floor(numbers.length * 0.12)));

    if (smallClassLikeTarget) {
      return {
        problemType: "classification",
        rationale: `The target "${targetColumn}" is numeric but has only ${uniqueNumbers.length} repeated class-like values.`,
      };
    }

    return {
      problemType: "regression",
      rationale: `The target "${targetColumn}" is numeric with many possible values, so the task is regression.`,
    };
  }

  function buildPreprocessor(rows, headers, targetColumn, problemType) {
    const supervised = problemType === "classification" || problemType === "regression";
    const usableRows = rows.filter((row) => {
      if (!supervised) return true;
      if (problemType === "regression") return Number.isFinite(toNumber(row[targetColumn]));
      return !isMissing(row[targetColumn]);
    });
    const featureColumns = headers.filter((header) => header !== targetColumn);
    const types = inferColumnTypes(usableRows, featureColumns);
    const numericColumns = featureColumns.filter((column) => types[column].kind === "numeric");
    const categoricalColumns = featureColumns.filter((column) => types[column].kind !== "numeric");
    const numericStats = {};
    const categoricalStats = {};
    const featureNames = [];
    let outlierCount = 0;

    numericColumns.forEach((column) => {
      const values = usableRows.map((row) => toNumber(row[column])).filter(Number.isFinite).sort((a, b) => a - b);
      const median = quantile(values, 0.5) ?? 0;
      const q1 = quantile(values, 0.25) ?? median;
      const q3 = quantile(values, 0.75) ?? median;
      const iqr = q3 - q1;
      const lower = q1 - 1.5 * iqr;
      const upper = q3 + 1.5 * iqr;
      const clipped = usableRows.map((row) => {
        const value = Number.isFinite(toNumber(row[column])) ? toNumber(row[column]) : median;
        if (value < lower || value > upper) outlierCount += 1;
        return clamp(value, lower, upper);
      });
      const mean = average(clipped);
      const std = standardDeviation(clipped, mean) || 1;
      numericStats[column] = {
        median,
        q1,
        q3,
        lower,
        upper,
        mean,
        std,
        missing: types[column]?.missing || 0,
      };
      featureNames.push(column);
    });

    categoricalColumns.forEach((column) => {
      const counts = countValues(usableRows.map((row) => (isMissing(row[column]) ? "__missing__" : String(row[column]).trim())));
      const mode = Object.entries(counts).sort((a, b) => b[1] - a[1])[0]?.[0] || "__missing__";
      const categories = Object.entries(counts)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 12)
        .map(([value]) => value);
      if (!categories.includes("__other__")) categories.push("__other__");
      categoricalStats[column] = {
        mode,
        categories,
        missing: types[column]?.missing || 0,
      };
      categories.forEach((category) => featureNames.push(`${column} = ${category}`));
    });

    function transformRow(row) {
      const vector = [];
      numericColumns.forEach((column) => {
        const stats = numericStats[column];
        const raw = Number.isFinite(toNumber(row[column])) ? toNumber(row[column]) : stats.median;
        const clipped = clamp(raw, stats.lower, stats.upper);
        vector.push((clipped - stats.mean) / stats.std);
      });
      categoricalColumns.forEach((column) => {
        const stats = categoricalStats[column];
        const raw = isMissing(row[column]) ? stats.mode : String(row[column]).trim();
        const category = stats.categories.includes(raw) ? raw : "__other__";
        stats.categories.forEach((knownCategory) => vector.push(knownCategory === category ? 1 : 0));
      });
      return vector;
    }

    const X = usableRows.map(transformRow);
    let y = null;
    let targetEncoder = null;

    if (problemType === "classification") {
      const classes = Array.from(new Set(usableRows.map((row) => String(row[targetColumn]).trim()))).sort();
      const classToIndex = Object.fromEntries(classes.map((label, index) => [label, index]));
      y = usableRows.map((row) => classToIndex[String(row[targetColumn]).trim()]);
      targetEncoder = { classes, classToIndex };
    } else if (problemType === "regression") {
      y = usableRows.map((row) => toNumber(row[targetColumn]));
    }

    return {
      X,
      y,
      rows: usableRows,
      featureColumns,
      targetColumn,
      problemType,
      numericColumns,
      categoricalColumns,
      numericStats,
      categoricalStats,
      featureNames,
      outlierCount,
      missingHandled: featureColumns.reduce((sum, column) => sum + (types[column]?.missing || 0), 0),
      targetEncoder,
      transformRow,
    };
  }

  function preprocessingSummary(processor) {
    const summary = [
      {
        title: "Missing values",
        body:
          processor.missingHandled === 0
            ? "No missing feature values were found."
            : `${processor.missingHandled} missing feature value(s) were imputed. Numeric columns used the median; categorical columns used the mode.`,
      },
      {
        title: "Outliers",
        body:
          processor.outlierCount === 0
            ? "No strong numeric outliers were detected by the IQR rule."
            : `${processor.outlierCount} numeric value(s) were capped using the IQR lower and upper fences so extreme points do not dominate training.`,
      },
      {
        title: "Encoding",
        body:
          processor.categoricalColumns.length === 0
            ? "No categorical feature encoding was needed."
            : `${processor.categoricalColumns.length} categorical column(s) were one-hot encoded so algorithms can learn from labels.`,
      },
      {
        title: "Scaling",
        body:
          processor.numericColumns.length === 0
            ? "No numeric feature scaling was needed."
            : `${processor.numericColumns.length} numeric column(s) were standardized to mean 0 and standard deviation 1.`,
      },
    ];

    if (processor.problemType === "classification") {
      summary.push({
        title: "Target encoding",
        body: `The target was converted into class ids: ${processor.targetEncoder.classes.join(", ")}.`,
      });
    }

    return summary;
  }

  function targetSummary(rows, targetColumn, problemType) {
    if (!targetColumn || problemType === "unsupervised") {
      return { labels: [], values: [], numericValues: [], caption: "No target column" };
    }
    const values = rows.map((row) => row[targetColumn]).filter((value) => !isMissing(value));
    if (problemType === "regression") {
      const numericValues = values.map(toNumber).filter(Number.isFinite);
      return { labels: [], values: [], numericValues, caption: `${numericValues.length} numeric target values` };
    }
    const counts = countValues(values.map((value) => String(value).trim()));
    const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]);
    return {
      labels: entries.map(([label]) => label),
      values: entries.map(([, count]) => count),
      numericValues: [],
      caption: `${entries.length} target class(es)`,
    };
  }

  function countValues(values) {
    return values.reduce((counts, value) => {
      counts[value] = (counts[value] || 0) + 1;
      return counts;
    }, {});
  }

  function quantile(sortedValues, q) {
    if (!sortedValues.length) return null;
    const position = (sortedValues.length - 1) * q;
    const lower = Math.floor(position);
    const upper = Math.ceil(position);
    const weight = position - lower;
    return sortedValues[lower] * (1 - weight) + sortedValues[upper] * weight;
  }

  function average(values) {
    return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
  }

  function standardDeviation(values, mean = average(values)) {
    if (values.length <= 1) return 0;
    const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
    return Math.sqrt(variance);
  }

  function clamp(value, lower, upper) {
    if (!Number.isFinite(lower) || !Number.isFinite(upper) || lower === upper) return value;
    return Math.min(upper, Math.max(lower, value));
  }

  window.MLData = {
    parseCSV,
    inferColumnTypes,
    profileDataset,
    inferProblemType,
    buildPreprocessor,
    preprocessingSummary,
    targetSummary,
    toNumber,
    isMissing,
    countValues,
    average,
    standardDeviation,
    quantile,
  };
})();
