# Distributed Classifier
This application is responsible for analysing evaluation results in `../evaluation/tests/` folder. 

## Command Line
```
usage: main.py [-h] [-p {regression,classification,datasets}]
               [-e {regression,classification}]
               [-a {regression,classification}]
               [-r {regression,classification}]
               [-t {regression,classification}] [-i {regression}]
               [-g {bp-ranking,bp-performance}] [-n {ward,average,complete}]
               [-s {true,false}] [-c CLUSTER_ANALYSIS] [-gg {method-dataset}]
               [-hst HIST] [-hm HEATMAP] [-pt PARTITION]

optional arguments:
  -h, --help            show this help message and exit
  -p {regression,classification,datasets}, --process {regression,classification,datasets}
                        Create data sets for evaluation.
  -e {regression,classification}, --evaluate {regression,classification}
                        Type of evaluation (regression or classification).
                        Evaluate data sets.
  -a {regression,classification}, --analysis {regression,classification}
                        Type of analysis (regression or classification).
  -r {regression,classification}, --rank {regression,classification}
                        Type of ranking (regression or classification).
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
  -gg {method-dataset}, --ggplot {method-dataset}
                        Create a specified graphic.
  -hst HIST, --hist HIST
                        Create a specified graphic.
  -hm HEATMAP, --heatmap HEATMAP
                        Create a specified graphic.
  -pt PARTITION, --partition PARTITION
                        Run all analysis about vertical partitions.
```

## Note
In development.