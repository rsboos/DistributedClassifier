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
        self.visible_trees_path = path.join(self.trees_path, 'pdf/')
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
            if len(metadata) > 2:
                name = 'arb{}_{}'.format(metadata[1], '_'.join(metadata[2:]))

        return name

    @staticmethod
    def human_readable_methods():
        return {
            "vote_plurality": "Plurality",
            "scf_simpson": "Social Choice Function Simpson",
            "scf_dowdall": "Social Choice Function Dowdall",
            "scf_copeland": "Social Choice Function Copeland",
            "scf_borda": "Social Choice Function Borda",
            "math_median": "Arithmetic-based Median",
            "math_mean": "Arithmetic-based Mean",
            "cmb_svc": "Combiner SVC",
            "cmb_mlp": "Combiner MLP",
            "cmb_knn": "Combiner KNN",
            "cmb_gnb": "Combiner GNB",
            "cmb_dtree": "Combiner DTREE",
            "classif_svc": "Base Classifier SVC",
            "classif_mlp": "Base Classifier MLP",
            "classif_knn": "Base Classifier KNN",
            "classif_gnb": "Base Classifier GNB",
            "classif_dtree": "Base Classifier DTREE",
            "arbmdic_svc": "Arbiter MDIC SVC",
            "arbmdic_mlp": "Arbiter MDIC MLP",
            "arbmdic_knn": "Arbiter MDIC KNN",
            "arbmdic_gnb": "Arbiter MDIC GNB",
            "arbmdic_dtree": "Arbiter MDIC DTREE",
            "arbmdi_svc": "Arbiter MDI SVC",
            "arbmdi_mlp": "Arbiter MDI MLP",
            "arbmdi_knn": "Arbiter MDI KNN",
            "arbmdi_gnb": "Arbiter MDI GNB",
            "arbmdi_dtree": "Arbiter MDI DTREE",
            "arbmd_svc": "Arbiter MD SVC",
            "arbmd_mlp": "Arbiter MD MLP",
            "arbmd_knn": "Arbiter MD KNN",
            "arbmd_gnb": "Arbiter MD GNB",
            "arbmd_dtree": "Arbiter MD DTREE",
            'arbmd': 'Arbiter MD',
            'arbmdi': 'Arbiter MDI',
            'arbmdic': 'Arbiter MDIC',
            'classif': 'Base classifiers',
            'cmb': 'Combiners',
            'math': 'Arithmetic-based',
            'scf': 'Social Choice Functions',
            'vote': 'Plurality'
        }

    @staticmethod
    def human_readable_types():
        return Path.human_readable_methods()

    @staticmethod
    def fix_method_name(method):
        method = method.split('/')[-1]
        method = method.replace('cv_scores_', '')
        method = method.replace('.csv', '')
        method = method.replace('_f1', '')

        if method.startswith('arb_'):
            method_p = method.split('_')
            if method_p[1] not in ['md', 'mdi', 'mdic']:
                method = "{s[0]}_{s[2]}_{s[1]}".format(s=method_p)

        return method


class RegressionPath(Path):

    def __init__(self):
        super().__init__('regression')


class ClassificationPath(Path):

    def __init__(self):
        super().__init__('classification')
