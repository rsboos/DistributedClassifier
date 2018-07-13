# Distributed Classifier
This application is responsible for analysing evaluation results in `../evaluation/tests/` folder. 

## Command Line
```
usage: main.py [-h] [-p PROCESS] [-e EVALUATE] [-t TREES] [-i IMPORTANT_NODES]
               [-a ALL]

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
  -i IMPORTANT_NODES, --get-important-nodes IMPORTANT_NODES
                        Only for regression. Extract important nodes from
                        trees.
  -a ALL, --all ALL     Make a pipeline with all analysis for regression or
                        classification.
```

## Note
In development.