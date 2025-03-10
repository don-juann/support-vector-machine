# Practical application of Support Vector Machines

## Overview
This notebook showcases use of SVM for different purposes using popular datasets.

Such as Fisher's Iris dataset, Wine dataset and California Housing dataset.

## LinearSVC, SVC, SGDClassifier on Iris dataset
Retrieving Data, separating data to make it suitable for binary classification:
```
iris = load_iris(as_frame=True)

# X containing Petal length and width
X = iris.data[['petal length (cm)', 'petal width (cm)']]

# y containing True or False labels for Setosa
y = (iris.target == 0)
```

Preprocessing data and creating instance of each model:
```
linsvm_clf = make_pipeline(
    StandardScaler(),
    LinearSVC(random_state=42)
)

svm_clf = make_pipeline(
    StandardScaler(),
    SVC(random_state=42)
)

sgd_clf = make_pipeline(
    StandardScaler(),
    SGDClassifier(random_state=42)
)
```

Testing model and getting results:
- LinearSVC accuracy: 1.0
- SVC accuracy: 1.0
- SGDClassifier accuracy: 1.0

## SVM Classifier on Wine dataset
Retrieving Data:
```
wine = load_wine(as_frame=True)
X = wine.data
y = wine.target
```

Preprocessing data and creating One-v-All (One-v-Rest) SVM model:
```
svm_clf = make_pipeline(
    StandardScaler(),
    SVC(kernel='rbf', decision_function_shape='ovr',random_state=42) # One-versus-Rest
)
```

Testing model and getting results:
- Accuracy: 1.0

## SVM Regression fine-tuned on California Housing dataset
Retrieving Data:
```
housing = fetch_california_housing(as_frame=True)

X = housing.data
y = housing.target
```

Creating parameter grid to fine-tune and find the best SVM model:
```
param_grid = {
    'kernel': ('poly', 'rbf'),
    'gamma': ('auto', 'scale'),
    'epsilon': (0.01, 0.1, 1),
    'C': (0.1, 1, 10)
}
```

Recreating the best model:
```
best_svr = SVR(C=1, epsilon=0.1, gamma='scale', kernel='rbf')
```

Testing model and getting results:
- MAE: 0.4
- RMSE: 0.59

## Results
1. All models performed great, meaning all three of them are capable of setting correct decision boundary for linear dataset. (Iris dataset with binary output in this case)
2. Model shows perfect score on test dataset. SVM classifiers are binary, but inputting several classes will automatically make it one-v-all classifier or we can explicitly specify it using 'decision_function_shape = ovr'.
3. Mean absolute error is around 40.000 U.S. dollars, while root mean squared error is about 60,000 U.S. dollars which is reasonable for this type of dataset and thus can be considered acceptable.

## Acknowledgements
Datasets are provided by sklearn.datasets

- Zhan Kazikhanov
- jkazikhanov@gmail.com
- https://github.com/don-juann
