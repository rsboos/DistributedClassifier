from pandas import DataFrame
from glob import glob


path = 'results/trees/*.dot'
trees = glob(path)

# Get nodes
labels = {}
for tree in trees:

    tree_name = tree.split('/')[-1][:-4]
    tree_file = open(tree)
    last_node = ''
    first_nodes = []
    nodes = []

    tree_file.readline()
    tree_file.readline()

    nodes.append(tree_file.readline())
    first_nodes.append(nodes[0])

    for node in tree_file:

        if node.startswith('0 -> '):
            i = int(node.split(' -> ')[1].split(' ')[0])

            first_nodes.append(nodes[i])

            if len(first_nodes) == 3:
                break
        elif ' -> ' not in node:
            nodes.append(node)

    tree_file.close()

    # Parse nodes
    labels[tree_name] = []
    for node in first_nodes:

        node_parts = node.split('"')
        node_parts = node_parts[1].split(' ')
        node_label = node_parts[0]

        labels[tree_name].append(node_label)

df = DataFrame(labels)
df = df.T.sort_index()
df.to_csv('results/tree_ranks/first_nodes.csv')
