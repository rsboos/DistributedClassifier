from os import path


class Path:

    def __init__(self, main_folder):
        self.default_file = 'cv_summary.csv'

        self.cluster_analysis = 'tests/cluster_analysis'
        self.test_path = path.join('tests/', main_folder)
        self.data_path = path.join(self.test_path, 'data/')

        self.results_path = path.join(self.test_path, 'results/')
        self.evaluation_path = path.join(self.results_path, 'evaluation/')

        self.trees_path = path.join(self.results_path, 'trees/')
        self.text_trees_path = path.join(self.trees_path, 'dots/')
        self.visible_trees_path = path.join(self.trees_path, 'png/')
        self.object_trees_path = path.join(self.trees_path, 'pkl/')

        self.graphics_path = path.join(self.test_path, 'graphics/')

    @staticmethod
    def concat_method_type(name):
        method_types = {'borda': 'scf',
                        'copeland': 'scf',
                        'dowdall': 'scf',
                        'simpson': 'scf',
                        'dtree': 'classif',
                        'gnb': 'classif',
                        'knn': 'classif',
                        'mlp': 'classif',
                        'svc': 'classif',
                        'mean': 'math',
                        'median': 'math',
                        'plurality': 'vote'}

        if name in method_types:
            return '{}_{}'.format(method_types[name], name)

        if name.startswith('arb'):
            metadata = name.split('_')
            name = 'arb{}_{}'.format(metadata[1], '_'.join(metadata[2:]))

        return name


class RegressionPath(Path):

    def __init__(self):
        super().__init__('regression')


class ClassificationPath(Path):

    def __init__(self):
        super().__init__('classification')
