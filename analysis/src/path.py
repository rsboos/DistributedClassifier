from os import path


class Path:
    default_file = 'cv_summary.csv'
    test_path = 'tests/regression'
    data_path = path.join(test_path, 'data/')
    results_path = path.join(test_path, 'results/')
    evaluation_path = path.join(results_path, 'evaluation/')
    trees_path = path.join(results_path, 'trees/')
    text_trees_path = path.join(trees_path, 'dots/')
    visible_trees_path = path.join(trees_path, 'png/')
    object_trees_path = path.join(trees_path, 'pkl/')
    graphics_path = path.join(test_path, 'graphics/')
