from pandas import read_csv, concat

ranks = read_csv('results/tree_ranks/ranks.csv', index_col=0)
firsts = read_csv('results/tree_ranks/first_nodes.csv', index_col=0)

ranks = ranks.iloc[:, 0:3]

comparison = ranks == firsts
accuracy = comparison.sum(axis=1) / 3

df = concat([comparison, accuracy], axis=1)
df.columns = ['0', '1', '2', 'accuracy']
df.to_csv('results/tree_ranks/comparison.csv')
