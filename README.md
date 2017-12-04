# Distributed Classifier
A distributed classifier for binary and multiclass classification problems.

Research project of Federal University of Rio Grande do Sul (UFRGS).

## Usage
```bash
usage: main.py [-h] -p PARAMS_FOLDER

optional arguments:
  -h, --help            show this help message and exit
  -p PARAMS_FOLDER, --params PARAMS_FOLDER
                        Folder where a file params.json is.
```

## Params file
A JSON file as follow:
```javascript
{
    // Dataset filepath
    "dataset": "datasets/seed.csv",

    // Class' column in dataset (default -1)
    // -1 for last column, 0 for first column
    "class_column": -1,

    // Number of cross-validation iterations (default 10)
    "iterations": 10,

    // Number of folds for cross-validation (default 10)
    "k_fold": 10,

    // Percentage of test instances from dataset
    // 0.2 means 20% of instances
    "test_size": 0.2,

    // Overlaped features
    // If float, should be between 0.0 and 1.0 and represents the percentage
    // of parts' common features. If int, should be less than or equal to the
    // number of features and represents the number of common features. If list,
    // represents the features' columns indexes. By default, the value is set to 0.
    "overlap": 0,

    // Classifiers
    // {<classifier id>: <full method call, i.e., with parameters>}
    "classifiers": {
        "svc": "sklearn.svm.SVC()"
    },

    // Scorer functions
    // {<scorer id>: <method call with parameters>}
    "metrics": {
        "auc": "sklearn.metrics.roc_auc_score()",
        "f1": "sklearn.metrics.f1_score(average='binary')",
        "recall": "sklearn.metrics.recall_score()",
        "specificity": "src.metrics.specificity_score()",
        "sensitivity": "src.metrics.sensitivity_score()",
        "accuracy": "sklearn.metrics.accuracy_score()",
        "precision": "sklearn.metrics.precision_score()"
    }
}
```

## Example
```bash
python3 main.py -p tests/test01
```
The program will save the **results** in the same folder.

## Results
Result files saved in *test folder*. You can find examples in `tests` folder.
- **cv_scores_\<classifier\>.csv**: scores for each Cross-Validation's iteration
- **model_\<classifier\>.pkl**: a pickel file to persist the created models
- **test_scores.csv**: testset's prediction scores
- **cv_summary.csv**: average scores from all *cv_scores_\<classifier\>.csv*

## Sample Datasets
This project uses a set of data samples for testing. This datasets are in `datasets/` folder.
