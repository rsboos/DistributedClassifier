# Distributed Classifier
This application is responsible for analysing evaluation results in `../evaluation/tests/` folder. 

## Command Line
```
usage: main.py [-h] [-p PROCESS] [-e EVALUATE] [-t TREES]

optional arguments:
  -h, --help            show this help message and exit
  -p PROCESS, --process PROCESS
                        Type of process (regression or classification). Create
                        data sets for evaluation.
  -e EVALUATE, --evaluate EVALUATE
                        Type of evaluation (regression or classification).
                        Evaluate data sets.
  -t TREES, --make-trees TREES
                        Only for regression. Create trees from
                        DecisionTreeRegressor.

```

## Note
In development.