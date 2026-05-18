# ML Concept Studio

An educational machine learning website that lets a learner upload a CSV dataset, diagnose whether the task is classification, regression, or unsupervised clustering, preprocess the data, visualize it, choose an algorithm, train it, and explain a single prediction.

## Languages Used

- HTML for the page structure
- CSS for the responsive dashboard design
- Vanilla JavaScript for CSV parsing, preprocessing, visualization, algorithms, metrics, and prediction explanations

This version intentionally avoids backend dependencies so it is easy to open and study in VS Code. A production version could move the ML pipeline to Python with Flask/FastAPI and scikit-learn.

## Project File Structure

```text
Machine Learning web/
├── index.html
├── README.md
├── assets/
│   ├── css/
│   │   └── styles.css
│   └── js/
│       ├── app.js
│       ├── charts.js
│       ├── data.js
│       ├── explanations.js
│       ├── ml.js
│       └── samples.js
└── sample-data/
    ├── customer_segments_clustering.csv
    ├── house_price_regression.csv
    └── student_success_classification.csv
```

## How To Run In VS Code

1. Open this folder in VS Code.
2. Open `index.html`.
3. Right-click the file and choose `Open with Live Server` if you use the Live Server extension.
4. You can also open `index.html` directly in a browser because the app has no build step.

## What The App Does

1. Uploads or loads a CSV dataset.
2. Lets you choose a target column, or choose no target for clustering.
3. Detects the learning type:
   - classification when the target is categorical or class-like
   - regression when the target is continuous numeric
   - unsupervised when no target is selected
4. Preprocesses features:
   - missing numeric values use median imputation
   - missing categorical values use mode imputation
   - numeric outliers are capped using the IQR rule
   - categorical features are one-hot encoded
   - numeric features are standardized
5. Visualizes missing values, target distribution, feature types, and numeric distributions.
6. Explains the selected algorithm before training.
7. Trains an educational model and shows metrics.
8. Predicts one row and explains why the output was selected.

## Algorithms Included

Classification:

- Logistic Regression
- K-Nearest Neighbors
- Naive Bayes
- Decision Tree
- Random Forest

Regression:

- Linear Regression
- KNN Regression
- Decision Tree Regression
- Random Forest Regression

Unsupervised:

- K-Means Clustering
- DBSCAN

## Learning Note

The algorithms are implemented in JavaScript for concept learning. They are good for understanding the pipeline and experimenting with small-to-medium CSV files. For larger or production-grade ML, use Python with pandas and scikit-learn.
