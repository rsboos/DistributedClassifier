from .data import Data
from .metrics import summary
from sklearn.metrics import make_scorer
from .simulator import FeatureDistributedSimulator
from .agents import ArbiterMetaDiff, ArbiterMetaDiffInc, ArbiterMetaDiffIncCorr


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


def load_arbiters(arbiters):
    methods = load_imports(arbiters['methods'])
    arbiter_objs = []

    for arbiter_class in arbiters['classes']:
        arb = eval(arbiter_class + '(methods)')
        arbiter_objs.append(arb)

    return arbiter_objs

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

        arbiters: list
            A list of Arbiter objects.

        combiner: Combiner
            A Combiner object

        mathematician: Mathematician
            A Mathematician object.

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
    arbiters = kwargs['arbiters']
    combiner = kwargs['combiner']
    mathematician = kwargs['mathematician']
    aggregators = [voter, combiner] + arbiters + [mathematician]

    # For results
    names = kwargs['names']
    results_path = kwargs['results_path']

    # Simulate distribution
    data = Data.load(filepath, class_column)

    # Create simulator (agents' manager)
    simulator = FeatureDistributedSimulator(data, classifiers, aggregators)

    # Cross validate
    ranks, scores = simulator.evaluate(overlap, random_state, scorers, iterations)

    # Save CV scores
    n = len(names)

    diff = n - len(scores)

    if diff > 0:
        nc = len(classifiers)
        names = names[0:nc - diff] + names[nc:]
        n = len(names)

    [scores[i].to_csv('{}/cv_scores_{}.csv'.format(results_path, names[i])) for i in range(n)]

    # Create CV summary
    stats = summary(scores)
    stats.index = names
    stats.to_csv('{}/cv_summary.csv'.format(results_path))
