from src.data import Data
from src.metrics import summary
from theobserver import Observer
from sklearn.metrics import make_scorer
from src.simulator import FeatureDistributed


def load_imports(imports):
    objs = list()

    for c in imports.values():                           # for each classfier
        i, modules, obj = split_parts(c)                 # separate in parts

        str_eval = "from {} import {}".format(modules, obj[:i])

        # Execute import
        exec(str_eval)

        # Evaluate a classifier
        objs.append(eval(obj))

    return objs


def split_parts(label):
    parts = label.split('.')                         # separate in parts

    i = ['(' in part for part in parts].index(1)     # filter parts

    modules = '.'.join(parts[:i])                    # get modules
    obj = '.'.join(parts[i:])                        # get obj
    i = obj.find('(')                                # filter obj

    return i, modules, obj


def load_scorers(metrics):
    scorers = dict()

    for n, c in metrics.items():                         # for each metric (name, method)
        i, modules, metric = split_parts(c)              # separate in parts

        params = metric[i + 1:-1]                        # get score func params (without '()')
        metric = metric[:i]                              # get score func name

        str_eval_import = "from {} import {}".format(modules, metric)
        str_eval_scorer = "make_scorer({}, {})".format(metric, params)

        # Execute import
        exec(str_eval_import)

        # Evaluate a metric
        scorers[n] = eval(str_eval_scorer)

    return scorers

def test(**kwargs):
    """Test over params and save results.

    Arguments
        overlap: float
            % of overlaped features.

        filepath: string
            Dataset's filepath.

        iterations: int
            Number of CV iterations.

        class_column: {-1, 0}
            When -1, class's column is the last one. When 0, first column is used.

        random_state: int
            Any number for generate CV folds.

        scorers: list
            A list of scorers returned by make_scorer.

        classifiers: list
            A list of classifiers objects.

        voter: Voter
            A Voter object.

        arbiter: Arbiter
            A Arbiter object.

        combiner: Combiner
            A Combiner object

        mathematician: Mathematician
            A Mathematician object.

        selectors: list
            A list of selectors.

        names: list
            Classifiers and aggregators' names.

        results_path: string
            Results' directory absolute/relative path.
    """

    # Data information
    overlap = kwargs['overlap']
    filepath = kwargs['filepath']
    iterations = kwargs['iterations']
    class_column = kwargs['class_column']
    random_state = kwargs['random_state']

    # Classifiers
    scorers = kwargs['scorers']
    classifiers = kwargs['classifiers']

    # Aggregators
    voter = kwargs['voter']
    arbiter = kwargs['arbiter']
    combiner = kwargs['combiner']
    selectors = kwargs['selectors']
    mathematician = kwargs['mathematician']

    # For results
    names = kwargs['names']
    results_path = kwargs['results_path']

    # Simulate distribution
    data = Data.load(filepath, class_column)

    # Create simulator (agents' manager)
    simulator = FeatureDistributed.load(data,
                                        classifiers,
                                        overlap,
                                        voter=voter,
                                        arbiter=arbiter,
                                        combiner=combiner,
                                        mathematician=mathematician)

    # Cross validate
    ranks, scores = simulator.repeated_cv(scorers, random_state, iterations)

    # Save CV scores
    n = len(names)
    [scores[i].to_csv('{}/cv_scores_{}.csv'.format(results_path, names[i])) for i in range(n)]

    # Create CV summary
    stats = summary(scores)
    stats.index = names
    stats.to_csv('{}/cv_summary.csv'.format(results_path))

    # Save dataset information
    obs = Observer(filepath, class_column)

    characteristics = obs.extract()
    characteristics += [len(classifiers), overlap, filepath]
    characteristics = list(map(lambda x: str(x), characteristics))

    file = open('tests/observer_data.csv', 'a+')
    file.seek(0)

    if all([filepath != line.split(',')[-1][:-1] for line in file]):
        file.write(','.join(characteristics) + '\n')

    file.close()
