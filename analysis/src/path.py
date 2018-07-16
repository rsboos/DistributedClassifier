from os import path


class Path:

    def __init__(self, main_folder):
        self.default_file = 'cv_summary.csv'

        self.test_path = path.join('tests/', main_folder)
        self.data_path = path.join(self.test_path, 'data/')

        self.results_path = path.join(self.test_path, 'results/')
        self.evaluation_path = path.join(self.results_path, 'evaluation/')

        self.trees_path = path.join(self.results_path, 'trees/')
        self.text_trees_path = path.join(self.trees_path, 'dots/')
        self.visible_trees_path = path.join(self.trees_path, 'png/')
        self.object_trees_path = path.join(self.trees_path, 'pkl/')

        self.graphics_path = path.join(self.test_path, 'graphics/')


class RegressionPath(Path):

    def __init__(self):
        super().__init__('regression')


class ClassificationPath(Path):

    def __init__(self):
        super().__init__('classification')
