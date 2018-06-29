from pandas import read_csv, DataFrame
from glob import glob

better_models = {}
for result in glob('results/evaluation/*'):
    method_name = result.split('/')[-1].split('.')[0]

    data = read_csv(result, header=[0, 1], index_col=0)
    data = data.loc[:, 'mean'].loc[:, 'mean_square']

    better_models[method_name] = data.idxmin()

dataframe = {'methods': list(better_models.keys()), 'better_regressor': list(better_models.values())}
DataFrame(dataframe).to_csv('results/models/better_models.csv', index=False)
