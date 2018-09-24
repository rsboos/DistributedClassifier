# Distributed Classifier
This application is responsible for analysing evaluation results in `../evaluation/tests/` folder. 

## Command Line
```
usage: main.py [-h] [-p {regression,classification}]
               [-e {regression,classification}]
               [-t {regression,classification}] [-i {regression}]
               [-g {bp-ranking,bp-performance}] [-n {ward,average,complete}]
               [-s {true,false}] [-c CLUSTER_ANALYSIS]

optional arguments:
  -h, --help            show this help message and exit
  -p {regression,classification}, --process {regression,classification}
                        Create data sets for evaluation.
  -e {regression,classification}, --evaluate {regression,classification}
                        Type of evaluation (regression or classification).
                        Evaluate data sets.
  -t {regression,classification}, --make-trees {regression,classification}
                        Create trees from DecisionTree's algorithm.
  -i {regression}, --get-important-nodes {regression}
                        Extract important nodes from trees.
  -g {bp-ranking,bp-performance}, --graphics {bp-ranking,bp-performance}
                        Create a specified graphic.
  -n {ward,average,complete}, --newick {ward,average,complete}
                        Display a Newick Tree.
  -s {true,false}, --show {true,false}
                        Show or not a graphic.
  -c CLUSTER_ANALYSIS, --cluster-analysis CLUSTER_ANALYSIS
                        Make a feature analysis by each cluster.
```

## Note
In development.