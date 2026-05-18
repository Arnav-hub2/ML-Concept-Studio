(function () {
  const ALGORITHMS = {
    classification: [
      {
        id: "logistic",
        name: "Logistic Regression",
        family: "Linear classifier",
        short: "Learns class probabilities by drawing weighted boundaries between classes.",
        concept:
          "Logistic regression is used for classification. It does not predict a raw number directly; it converts a weighted feature score into class probabilities and chooses the class with the highest probability.",
        advantages: [
          "Fast and easy to explain",
          "Gives useful probabilities",
          "Strong baseline when the classes are close to linearly separable",
        ],
        steps: [
          "Scale numeric features and one-hot encode categories",
          "Learn a weight for every feature and every class",
          "Convert class scores into probabilities",
          "Choose the class with the highest probability",
        ],
      },
      {
        id: "knnClassifier",
        name: "K-Nearest Neighbors",
        family: "Instance-based classifier",
        short: "Classifies a row by looking at the labels of the most similar training rows.",
        concept:
          "KNN keeps the training data in memory. For a new row, it measures distance to old rows and lets the nearest examples vote.",
        advantages: [
          "Simple mental model",
          "Works with non-linear patterns",
          "Good for small datasets with meaningful distances",
        ],
        steps: [
          "Preprocess every feature onto comparable scales",
          "Measure distance from the new row to training rows",
          "Take the nearest K rows",
          "Predict the most common class among the neighbors",
        ],
      },
      {
        id: "naiveBayes",
        name: "Naive Bayes",
        family: "Probabilistic classifier",
        short: "Scores each class using feature likelihoods and prior class frequency.",
        concept:
          "Naive Bayes estimates how likely the feature values are for each class. It is called naive because it treats features as mostly independent.",
        advantages: [
          "Very fast training",
          "Works surprisingly well on noisy data",
          "Useful when you need a probability-style explanation",
        ],
        steps: [
          "Calculate how often each class appears",
          "Estimate feature distributions inside each class",
          "Score every class for the new row",
          "Pick the class with the highest posterior score",
        ],
      },
      {
        id: "treeClassifier",
        name: "Decision Tree",
        family: "Rule-based classifier",
        short: "Builds if-then feature splits until rows become mostly one class.",
        concept:
          "A decision tree learns a sequence of questions. Each split tries to make the child groups purer than the parent group.",
        advantages: [
          "Easy to visualize as rules",
          "Captures non-linear relationships",
          "Handles mixed feature interactions well",
        ],
        steps: [
          "Try many feature thresholds",
          "Choose the split that lowers impurity the most",
          "Repeat on each branch",
          "Predict using the class distribution in the final leaf",
        ],
      },
      {
        id: "forestClassifier",
        name: "Random Forest",
        family: "Ensemble classifier",
        short: "Trains many decision trees and combines their votes.",
        concept:
          "A random forest reduces the instability of one decision tree by training many trees on different samples and feature subsets.",
        advantages: [
          "Usually more accurate than one tree",
          "Handles non-linear patterns",
          "Provides practical feature importance",
        ],
        steps: [
          "Create many bootstrapped training samples",
          "Train a decision tree on each sample",
          "Let all trees vote for the class",
          "Choose the class with the strongest vote",
        ],
      },
    ],
    regression: [
      {
        id: "linearRegression",
        name: "Linear Regression",
        family: "Linear regressor",
        short: "Predicts a continuous number from weighted feature contributions.",
        concept:
          "Linear regression starts from the average target value and adjusts the prediction up or down using learned feature weights.",
        advantages: [
          "Clear feature contribution story",
          "Fast baseline for numeric prediction",
          "Useful when relationships are roughly straight-line",
        ],
        steps: [
          "Prepare numeric and categorical features",
          "Scale the target internally for stable training",
          "Learn feature weights that reduce squared error",
          "Return the continuous predicted value",
        ],
      },
      {
        id: "knnRegressor",
        name: "KNN Regression",
        family: "Instance-based regressor",
        short: "Predicts by averaging the target values of the nearest rows.",
        concept:
          "KNN regression assumes similar rows should have similar target values. It looks nearby and averages what happened there.",
        advantages: [
          "Easy to understand",
          "Can follow curved relationships",
          "No heavy training phase",
        ],
        steps: [
          "Scale features so distance is fair",
          "Find the K most similar training rows",
          "Average their target values",
          "Use that average as the prediction",
        ],
      },
      {
        id: "treeRegressor",
        name: "Decision Tree Regression",
        family: "Rule-based regressor",
        short: "Splits rows into groups and predicts the average target inside the final group.",
        concept:
          "A regression tree learns if-then rules. Each split tries to create groups with lower target variance.",
        advantages: [
          "Captures non-linear patterns",
          "Creates readable rules",
          "Does not require a straight-line relationship",
        ],
        steps: [
          "Search feature thresholds",
          "Pick splits that reduce target variance",
          "Stop when branches are small or deep enough",
          "Predict the leaf average",
        ],
      },
      {
        id: "forestRegressor",
        name: "Random Forest Regression",
        family: "Ensemble regressor",
        short: "Averages predictions from many regression trees.",
        concept:
          "A random forest regressor combines many slightly different trees. Averaging them usually gives a steadier prediction than one tree.",
        advantages: [
          "Strong general-purpose model",
          "Handles feature interactions",
          "Less sensitive to one unlucky split",
        ],
        steps: [
          "Build bootstrapped samples",
          "Train several regression trees",
          "Predict with every tree",
          "Average the tree predictions",
        ],
      },
    ],
    unsupervised: [
      {
        id: "kmeans",
        name: "K-Means Clustering",
        family: "Centroid clustering",
        short: "Finds K groups by moving cluster centers toward nearby rows.",
        concept:
          "K-Means groups rows by distance. It repeatedly assigns rows to the nearest center and then moves each center to the average of its assigned rows.",
        advantages: [
          "Fast and intuitive",
          "Good first clustering model",
          "Produces clear cluster centers",
        ],
        steps: [
          "Choose the number of clusters",
          "Place initial centers",
          "Assign every row to the closest center",
          "Move centers and repeat until stable",
        ],
      },
      {
        id: "dbscan",
        name: "DBSCAN",
        family: "Density clustering",
        short: "Finds dense neighborhoods and marks isolated rows as noise.",
        concept:
          "DBSCAN does not require choosing a fixed number of clusters. It groups points that are packed closely together and treats lonely points as noise.",
        advantages: [
          "Can find irregular cluster shapes",
          "Detects outliers as noise",
          "No need to decide K before training",
        ],
        steps: [
          "Estimate a neighborhood radius",
          "Find rows with enough nearby neighbors",
          "Expand clusters from dense rows",
          "Leave isolated rows as noise",
        ],
      },
    ],
  };

  function getAlgorithms(problemType) {
    return ALGORITHMS[problemType] || [];
  }

  function getAlgorithm(problemType, id) {
    return getAlgorithms(problemType).find((algorithm) => algorithm.id === id);
  }

  window.MLExplanations = {
    ALGORITHMS,
    getAlgorithms,
    getAlgorithm,
  };
})();
