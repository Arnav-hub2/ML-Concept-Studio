(function () {
  function train(problemType, algorithmId, X, y, options = {}) {
    if (!X.length) throw new Error("No training rows are available.");
    if (problemType === "classification") {
      const classCount = options.classCount || Math.max(...y) + 1;
      if (algorithmId === "logistic") return trainSoftmaxRegression(X, y, classCount);
      if (algorithmId === "knnClassifier") return trainKnnClassifier(X, y, classCount);
      if (algorithmId === "naiveBayes") return trainNaiveBayes(X, y, classCount);
      if (algorithmId === "treeClassifier") return trainDecisionTreeModel(X, y, "classification", { classCount });
      if (algorithmId === "forestClassifier") return trainRandomForest(X, y, "classification", { classCount });
    }

    if (problemType === "regression") {
      if (algorithmId === "linearRegression") return trainLinearRegression(X, y);
      if (algorithmId === "knnRegressor") return trainKnnRegressor(X, y);
      if (algorithmId === "treeRegressor") return trainDecisionTreeModel(X, y, "regression");
      if (algorithmId === "forestRegressor") return trainRandomForest(X, y, "regression");
    }

    if (problemType === "unsupervised") {
      if (algorithmId === "kmeans") return trainKMeans(X, options.k || 3);
      if (algorithmId === "dbscan") return trainDbscan(X);
    }

    throw new Error(`Unsupported algorithm: ${algorithmId}`);
  }

  function splitDataset(X, y, testRatio = 0.25, seed = 11) {
    const random = mulberry32(seed);
    const indices = X.map((_, index) => index).sort(() => random() - 0.5);
    const testSize = Math.max(1, Math.floor(indices.length * testRatio));
    const testIndices = indices.slice(0, testSize);
    const trainIndices = indices.slice(testSize);
    return {
      trainX: trainIndices.map((index) => X[index]),
      trainY: y ? trainIndices.map((index) => y[index]) : null,
      testX: testIndices.map((index) => X[index]),
      testY: y ? testIndices.map((index) => y[index]) : null,
      trainIndices,
      testIndices,
    };
  }

  function evaluate(problemType, model, X, y, classLabels = []) {
    if (problemType === "regression") {
      const predictions = X.map((row) => model.predictOne(row));
      const meanY = average(y);
      const mae = average(predictions.map((prediction, index) => Math.abs(prediction - y[index])));
      const rmse = Math.sqrt(average(predictions.map((prediction, index) => (prediction - y[index]) ** 2)));
      const ssRes = predictions.reduce((sum, prediction, index) => sum + (prediction - y[index]) ** 2, 0);
      const ssTot = y.reduce((sum, actual) => sum + (actual - meanY) ** 2, 0) || 1;
      return { predictions, mae, rmse, r2: 1 - ssRes / ssTot };
    }

    if (problemType === "classification") {
      const predictions = X.map((row) => model.predictOne(row));
      const classCount = classLabels.length || Math.max(...y, ...predictions) + 1;
      const matrix = Array.from({ length: classCount }, () => Array.from({ length: classCount }, () => 0));
      predictions.forEach((prediction, index) => {
        matrix[y[index]][prediction] += 1;
      });
      const correct = predictions.filter((prediction, index) => prediction === y[index]).length;
      const f1Scores = matrix.map((_, classIndex) => {
        const tp = matrix[classIndex][classIndex];
        const fp = matrix.reduce((sum, row, rowIndex) => (rowIndex === classIndex ? sum : sum + row[classIndex]), 0);
        const fn = matrix[classIndex].reduce((sum, value, colIndex) => (colIndex === classIndex ? sum : sum + value), 0);
        const precision = tp + fp === 0 ? 0 : tp / (tp + fp);
        const recall = tp + fn === 0 ? 0 : tp / (tp + fn);
        return precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);
      });
      return {
        predictions,
        accuracy: correct / predictions.length,
        macroF1: average(f1Scores),
        matrix,
      };
    }

    return {};
  }

  function trainLinearRegression(X, y) {
    const n = X.length;
    const d = X[0].length;
    const yMean = average(y);
    const yStd = standardDeviation(y, yMean) || 1;
    const scaledY = y.map((value) => (value - yMean) / yStd);
    const weights = Array.from({ length: d + 1 }, () => 0);
    const learningRate = 0.04;
    const l2 = 0.001;
    const epochs = 900;

    for (let epoch = 0; epoch < epochs; epoch += 1) {
      const gradient = Array.from({ length: d + 1 }, () => 0);
      for (let i = 0; i < n; i += 1) {
        const prediction = dotWithBias(weights, X[i]);
        const error = prediction - scaledY[i];
        gradient[0] += error;
        for (let j = 0; j < d; j += 1) gradient[j + 1] += error * X[i][j];
      }
      for (let j = 0; j <= d; j += 1) {
        const penalty = j === 0 ? 0 : l2 * weights[j];
        weights[j] -= learningRate * (gradient[j] / n + penalty);
      }
    }

    const importances = weights.slice(1).map(Math.abs);
    return {
      kind: "linearRegression",
      weights,
      importances,
      predictOne(row) {
        return dotWithBias(weights, row) * yStd + yMean;
      },
      explainVector(row) {
        return weights.slice(1).map((weight, index) => weight * row[index] * yStd);
      },
      details: { yMean, yStd },
    };
  }

  function trainSoftmaxRegression(X, y, classCount) {
    const n = X.length;
    const d = X[0].length;
    const weights = Array.from({ length: classCount }, () => Array.from({ length: d + 1 }, () => 0));
    const learningRate = 0.08;
    const l2 = 0.001;
    const epochs = 520;

    for (let epoch = 0; epoch < epochs; epoch += 1) {
      const gradient = weights.map((row) => row.map(() => 0));
      for (let i = 0; i < n; i += 1) {
        const probs = softmax(weights.map((rowWeights) => dotWithBias(rowWeights, X[i])));
        for (let c = 0; c < classCount; c += 1) {
          const error = probs[c] - (y[i] === c ? 1 : 0);
          gradient[c][0] += error;
          for (let j = 0; j < d; j += 1) gradient[c][j + 1] += error * X[i][j];
        }
      }
      for (let c = 0; c < classCount; c += 1) {
        for (let j = 0; j <= d; j += 1) {
          const penalty = j === 0 ? 0 : l2 * weights[c][j];
          weights[c][j] -= learningRate * (gradient[c][j] / n + penalty);
        }
      }
    }

    const importances = Array.from({ length: d }, (_, featureIndex) =>
      weights.reduce((sum, classWeights) => sum + Math.abs(classWeights[featureIndex + 1]), 0)
    );

    return {
      kind: "softmaxRegression",
      weights,
      importances,
      predictProba(row) {
        return softmax(weights.map((rowWeights) => dotWithBias(rowWeights, row)));
      },
      predictOne(row) {
        return argMax(this.predictProba(row));
      },
      explainVector(row, classIndex) {
        const chosen = classIndex ?? this.predictOne(row);
        return weights[chosen].slice(1).map((weight, index) => weight * row[index]);
      },
    };
  }

  function trainKnnClassifier(X, y, classCount) {
    const k = Math.min(7, Math.max(3, Math.round(Math.sqrt(X.length))));
    return {
      kind: "knnClassifier",
      importances: varianceByFeature(X),
      predictOne(row) {
        return argMax(this.predictProba(row));
      },
      predictProba(row) {
        const neighbors = nearestNeighbors(X, y, row, k);
        const votes = Array.from({ length: classCount }, () => 0);
        neighbors.forEach((neighbor) => {
          votes[neighbor.label] += 1 / (neighbor.distance + 1e-6);
        });
        const total = votes.reduce((sum, value) => sum + value, 0) || 1;
        return votes.map((value) => value / total);
      },
      details: { k },
    };
  }

  function trainKnnRegressor(X, y) {
    const k = Math.min(7, Math.max(3, Math.round(Math.sqrt(X.length))));
    return {
      kind: "knnRegressor",
      importances: varianceByFeature(X),
      predictOne(row) {
        const neighbors = nearestNeighbors(X, y, row, k);
        const totalWeight = neighbors.reduce((sum, neighbor) => sum + 1 / (neighbor.distance + 1e-6), 0);
        return (
          neighbors.reduce((sum, neighbor) => sum + neighbor.label * (1 / (neighbor.distance + 1e-6)), 0) /
          (totalWeight || 1)
        );
      },
      details: { k },
    };
  }

  function trainNaiveBayes(X, y, classCount) {
    const d = X[0].length;
    const priors = Array.from({ length: classCount }, () => 0);
    const means = Array.from({ length: classCount }, () => Array.from({ length: d }, () => 0));
    const variances = Array.from({ length: classCount }, () => Array.from({ length: d }, () => 1));

    for (let c = 0; c < classCount; c += 1) {
      const rows = X.filter((_, index) => y[index] === c);
      priors[c] = rows.length / X.length || 1e-6;
      for (let j = 0; j < d; j += 1) {
        const values = rows.map((row) => row[j]);
        means[c][j] = average(values);
        variances[c][j] = Math.max(standardDeviation(values, means[c][j]) ** 2, 1e-4);
      }
    }

    const importances = Array.from({ length: d }, (_, featureIndex) => {
      const classMeans = means.map((row) => row[featureIndex]);
      return standardDeviation(classMeans, average(classMeans));
    });

    return {
      kind: "naiveBayes",
      importances,
      predictProba(row) {
        const logScores = priors.map((prior, classIndex) => {
          let score = Math.log(prior);
          for (let j = 0; j < d; j += 1) {
            const variance = variances[classIndex][j];
            const diff = row[j] - means[classIndex][j];
            score += -0.5 * Math.log(2 * Math.PI * variance) - (diff * diff) / (2 * variance);
          }
          return score;
        });
        return softmax(logScores);
      },
      predictOne(row) {
        return argMax(this.predictProba(row));
      },
    };
  }

  function trainDecisionTreeModel(X, y, task, options = {}) {
    const importances = Array.from({ length: X[0].length }, () => 0);
    const tree = buildTree(X, y, X.map((_, index) => index), task, {
      depth: 0,
      maxDepth: 6,
      minSamples: 4,
      classCount: options.classCount,
      importances,
    });

    return {
      kind: task === "classification" ? "treeClassifier" : "treeRegressor",
      tree,
      importances,
      predictOne(row) {
        return predictTree(tree, row);
      },
      predictProba(row) {
        const leaf = findLeaf(tree, row);
        return leaf.proba || [];
      },
    };
  }

  function trainRandomForest(X, y, task, options = {}) {
    const random = mulberry32(19);
    const treeCount = task === "classification" ? 17 : 15;
    const d = X[0].length;
    const importances = Array.from({ length: d }, () => 0);
    const trees = [];

    for (let treeIndex = 0; treeIndex < treeCount; treeIndex += 1) {
      const sampleIndices = Array.from({ length: X.length }, () => Math.floor(random() * X.length));
      const treeImportances = Array.from({ length: d }, () => 0);
      trees.push(
        buildTree(X, y, sampleIndices, task, {
          depth: 0,
          maxDepth: 7,
          minSamples: 4,
          classCount: options.classCount,
          featureSubset: Math.max(1, Math.round(Math.sqrt(d))),
          random,
          importances: treeImportances,
        })
      );
      treeImportances.forEach((value, index) => {
        importances[index] += value / treeCount;
      });
    }

    return {
      kind: task === "classification" ? "forestClassifier" : "forestRegressor",
      trees,
      importances,
      predictOne(row) {
        const predictions = trees.map((tree) => predictTree(tree, row));
        if (task === "regression") return average(predictions);
        const counts = {};
        predictions.forEach((prediction) => {
          counts[prediction] = (counts[prediction] || 0) + 1;
        });
        return Number(Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0]);
      },
      predictProba(row) {
        if (task !== "classification") return [];
        const votes = Array.from({ length: options.classCount }, () => 0);
        trees.forEach((tree) => {
          votes[predictTree(tree, row)] += 1;
        });
        return votes.map((vote) => vote / trees.length);
      },
    };
  }

  function buildTree(X, y, indices, task, options) {
    const { depth, maxDepth, minSamples, classCount, importances } = options;
    const currentImpurity = impurity(y, indices, task, classCount);
    if (depth >= maxDepth || indices.length <= minSamples || currentImpurity <= 1e-9) {
      return makeLeaf(y, indices, task, classCount);
    }

    const featureCount = X[0].length;
    const features = chooseFeatures(featureCount, options.featureSubset, options.random);
    let best = null;

    features.forEach((feature) => {
      const candidates = candidateThresholds(X, indices, feature);
      candidates.forEach((threshold) => {
        const left = [];
        const right = [];
        indices.forEach((index) => {
          if (X[index][feature] <= threshold) left.push(index);
          else right.push(index);
        });
        if (left.length < minSamples || right.length < minSamples) return;
        const leftImpurity = impurity(y, left, task, classCount);
        const rightImpurity = impurity(y, right, task, classCount);
        const weighted = (left.length / indices.length) * leftImpurity + (right.length / indices.length) * rightImpurity;
        const gain = currentImpurity - weighted;
        if (!best || gain > best.gain) best = { feature, threshold, left, right, gain };
      });
    });

    if (!best || best.gain <= 1e-9) return makeLeaf(y, indices, task, classCount);
    importances[best.feature] += best.gain * indices.length;

    return {
      feature: best.feature,
      threshold: best.threshold,
      left: buildTree(X, y, best.left, task, { ...options, depth: depth + 1 }),
      right: buildTree(X, y, best.right, task, { ...options, depth: depth + 1 }),
    };
  }

  function makeLeaf(y, indices, task, classCount) {
    if (task === "regression") {
      return { leaf: true, value: average(indices.map((index) => y[index])) };
    }
    const counts = Array.from({ length: classCount }, () => 0);
    indices.forEach((index) => {
      counts[y[index]] += 1;
    });
    const total = counts.reduce((sum, value) => sum + value, 0) || 1;
    return {
      leaf: true,
      value: argMax(counts),
      proba: counts.map((count) => count / total),
    };
  }

  function predictTree(node, row) {
    if (node.leaf) return node.value;
    return predictTree(row[node.feature] <= node.threshold ? node.left : node.right, row);
  }

  function findLeaf(node, row) {
    if (node.leaf) return node;
    return findLeaf(row[node.feature] <= node.threshold ? node.left : node.right, row);
  }

  function impurity(y, indices, task, classCount) {
    if (task === "regression") {
      const values = indices.map((index) => y[index]);
      const mean = average(values);
      return average(values.map((value) => (value - mean) ** 2));
    }
    const counts = Array.from({ length: classCount }, () => 0);
    indices.forEach((index) => {
      counts[y[index]] += 1;
    });
    const total = indices.length || 1;
    return 1 - counts.reduce((sum, count) => sum + (count / total) ** 2, 0);
  }

  function candidateThresholds(X, indices, feature) {
    const values = Array.from(new Set(indices.map((index) => X[index][feature]).filter(Number.isFinite))).sort((a, b) => a - b);
    if (values.length <= 1) return [];
    const thresholds = [];
    const step = Math.max(1, Math.floor(values.length / 12));
    for (let i = step; i < values.length; i += step) {
      thresholds.push((values[i - 1] + values[i]) / 2);
    }
    return thresholds;
  }

  function chooseFeatures(featureCount, subsetSize, random = Math.random) {
    const features = Array.from({ length: featureCount }, (_, index) => index);
    if (!subsetSize || subsetSize >= featureCount) return features;
    return features.sort(() => random() - 0.5).slice(0, subsetSize);
  }

  function trainKMeans(X, k) {
    const random = mulberry32(29);
    const points = X;
    const d = X[0].length;
    const centroids = Array.from({ length: k }, () => points[Math.floor(random() * points.length)].slice());
    let assignments = Array.from({ length: points.length }, () => 0);

    for (let iteration = 0; iteration < 80; iteration += 1) {
      let changed = false;
      assignments = points.map((point, index) => {
        const next = nearestCentroid(point, centroids);
        if (next !== assignments[index]) changed = true;
        return next;
      });
      const sums = Array.from({ length: k }, () => Array.from({ length: d }, () => 0));
      const counts = Array.from({ length: k }, () => 0);
      points.forEach((point, index) => {
        const cluster = assignments[index];
        counts[cluster] += 1;
        for (let j = 0; j < d; j += 1) sums[cluster][j] += point[j];
      });
      for (let cluster = 0; cluster < k; cluster += 1) {
        if (counts[cluster] === 0) continue;
        for (let j = 0; j < d; j += 1) centroids[cluster][j] = sums[cluster][j] / counts[cluster];
      }
      if (!changed) break;
    }

    const inertia = points.reduce((sum, point, index) => sum + squaredDistance(point, centroids[assignments[index]]), 0);
    return {
      kind: "kmeans",
      centroids,
      assignments,
      importances: varianceByFeature(X),
      predictOne(row) {
        return nearestCentroid(row, centroids);
      },
      details: { k, inertia },
    };
  }

  function trainDbscan(X) {
    const eps = estimateEps(X);
    const minPts = Math.min(5, Math.max(3, Math.round(Math.sqrt(X.length) / 2)));
    const labels = Array.from({ length: X.length }, () => undefined);
    let clusterId = 0;

    for (let index = 0; index < X.length; index += 1) {
      if (labels[index] !== undefined) continue;
      const neighbors = regionQuery(X, index, eps);
      if (neighbors.length < minPts) {
        labels[index] = -1;
        continue;
      }
      labels[index] = clusterId;
      const queue = neighbors.filter((neighbor) => neighbor !== index);
      while (queue.length) {
        const neighbor = queue.shift();
        if (labels[neighbor] === -1) labels[neighbor] = clusterId;
        if (labels[neighbor] !== undefined) continue;
        labels[neighbor] = clusterId;
        const expanded = regionQuery(X, neighbor, eps);
        if (expanded.length >= minPts) queue.push(...expanded);
      }
      clusterId += 1;
    }

    return {
      kind: "dbscan",
      assignments: labels,
      importances: varianceByFeature(X),
      predictOne(row) {
        let best = { label: -1, distance: Infinity };
        X.forEach((point, index) => {
          if (labels[index] === -1) return;
          const distance = euclideanDistance(point, row);
          if (distance < best.distance) best = { label: labels[index], distance };
        });
        return best.distance <= eps ? best.label : -1;
      },
      details: { eps, minPts, clusterCount: clusterId, noiseCount: labels.filter((label) => label === -1).length },
    };
  }

  function pca2D(X) {
    if (!X.length) return [];
    const d = X[0].length;
    if (d === 1) return X.map((row) => [row[0], 0]);
    const means = Array.from({ length: d }, (_, feature) => average(X.map((row) => row[feature])));
    const centered = X.map((row) => row.map((value, feature) => value - means[feature]));
    const covariance = Array.from({ length: d }, (_, i) =>
      Array.from({ length: d }, (_, j) => average(centered.map((row) => row[i] * row[j])))
    );
    const first = powerIteration(covariance);
    const lambda = multiplyMatrixVector(covariance, first).reduce((sum, value, index) => sum + value * first[index], 0);
    const deflated = covariance.map((row, i) => row.map((value, j) => value - lambda * first[i] * first[j]));
    const second = powerIteration(deflated);
    return centered.map((row) => [dot(row, first), dot(row, second)]);
  }

  function explainPrediction(problemType, model, row, processor, result, labels = []) {
    const featureNames = processor.featureNames;
    const importances = model.importances || [];
    const signals = featureNames
      .map((name, index) => {
        const contribution = model.explainVector ? Math.abs(model.explainVector(row, result?.classIndex)[index] || 0) : Math.abs(row[index]) * (importances[index] || 0);
        return { name, contribution, direction: row[index] >= 0 ? "above typical" : "below typical" };
      })
      .sort((a, b) => b.contribution - a.contribution)
      .slice(0, 4)
      .filter((signal) => signal.contribution > 0);

    if (problemType === "classification") {
      const probabilities = model.predictProba ? model.predictProba(row) : [];
      const ranked = probabilities
        .map((value, index) => ({ label: labels[index] ?? String(index), value }))
        .sort((a, b) => b.value - a.value);
      const winner = ranked[0];
      const runnerUp = ranked[1];
      return {
        headline: `Predicted class: ${winner?.label ?? labels[result] ?? result}`,
        why: winner
          ? `The selected class had the strongest model score at ${Math.round(winner.value * 100)}%.`
          : "The model selected the class with the strongest vote.",
        whyNot:
          runnerUp && winner
            ? `${runnerUp.label} was not selected because its score was lower at ${Math.round(runnerUp.value * 100)}%.`
            : "No competing class was close enough to replace the prediction.",
        signals,
      };
    }

    if (problemType === "regression") {
      return {
        headline: `Predicted value: ${formatNumber(result)}`,
        why:
          model.kind === "linearRegression"
            ? "The model started from the target average and adjusted the number using learned feature weights."
            : "The model compared this row with learned groups or neighbors and returned the most likely numeric value.",
        whyNot:
          "Regression models choose a continuous value, so there is no second class. A different value would need stronger evidence from the feature pattern.",
        signals,
      };
    }

    return {
      headline: `Assigned cluster: ${result === -1 ? "Noise / outlier" : result}`,
      why:
        result === -1
          ? "The row did not sit close enough to a dense learned group."
          : "The row was closest to the learned cluster pattern after preprocessing.",
      whyNot: "Other clusters were farther away in the standardized feature space.",
      signals,
    };
  }

  function nearestNeighbors(X, y, row, k) {
    return X.map((point, index) => ({ label: y[index], distance: euclideanDistance(point, row) }))
      .sort((a, b) => a.distance - b.distance)
      .slice(0, k);
  }

  function nearestCentroid(point, centroids) {
    let best = 0;
    let bestDistance = Infinity;
    centroids.forEach((centroid, index) => {
      const distance = squaredDistance(point, centroid);
      if (distance < bestDistance) {
        best = index;
        bestDistance = distance;
      }
    });
    return best;
  }

  function estimateEps(X) {
    const sample = X.slice(0, Math.min(220, X.length));
    const neighborDistances = sample.map((point, index) => {
      const distances = sample
        .map((other, otherIndex) => (index === otherIndex ? Infinity : euclideanDistance(point, other)))
        .sort((a, b) => a - b);
      return distances[Math.min(4, distances.length - 1)] || 1;
    });
    neighborDistances.sort((a, b) => a - b);
    return neighborDistances[Math.floor(neighborDistances.length * 0.65)] || 1;
  }

  function regionQuery(X, index, eps) {
    const neighbors = [];
    X.forEach((point, pointIndex) => {
      if (euclideanDistance(X[index], point) <= eps) neighbors.push(pointIndex);
    });
    return neighbors;
  }

  function powerIteration(matrix) {
    let vector = Array.from({ length: matrix.length }, (_, index) => (index === 0 ? 1 : 0.3));
    for (let iteration = 0; iteration < 80; iteration += 1) {
      vector = normalizeVector(multiplyMatrixVector(matrix, vector));
    }
    return vector;
  }

  function multiplyMatrixVector(matrix, vector) {
    return matrix.map((row) => dot(row, vector));
  }

  function normalizeVector(vector) {
    const magnitude = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0)) || 1;
    return vector.map((value) => value / magnitude);
  }

  function varianceByFeature(X) {
    if (!X.length) return [];
    return Array.from({ length: X[0].length }, (_, feature) => {
      const values = X.map((row) => row[feature]);
      return standardDeviation(values, average(values));
    });
  }

  function dotWithBias(weights, row) {
    let value = weights[0];
    for (let index = 0; index < row.length; index += 1) value += weights[index + 1] * row[index];
    return value;
  }

  function dot(a, b) {
    return a.reduce((sum, value, index) => sum + value * b[index], 0);
  }

  function softmax(scores) {
    const max = Math.max(...scores);
    const exps = scores.map((score) => Math.exp(score - max));
    const total = exps.reduce((sum, value) => sum + value, 0) || 1;
    return exps.map((value) => value / total);
  }

  function argMax(values) {
    let best = 0;
    for (let index = 1; index < values.length; index += 1) {
      if (values[index] > values[best]) best = index;
    }
    return best;
  }

  function euclideanDistance(a, b) {
    return Math.sqrt(squaredDistance(a, b));
  }

  function squaredDistance(a, b) {
    return a.reduce((sum, value, index) => sum + (value - b[index]) ** 2, 0);
  }

  function average(values) {
    return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
  }

  function standardDeviation(values, mean = average(values)) {
    if (values.length <= 1) return 0;
    return Math.sqrt(values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length);
  }

  function formatNumber(value) {
    if (!Number.isFinite(value)) return "-";
    if (Math.abs(value) >= 100) return value.toFixed(2).replace(/\.?0+$/, "");
    return value.toFixed(3).replace(/\.?0+$/, "");
  }

  function mulberry32(seed) {
    return function () {
      let t = (seed += 0x6d2b79f5);
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  window.MLModels = {
    train,
    splitDataset,
    evaluate,
    pca2D,
    explainPrediction,
  };
})();
