import argparse
import time
import numpy as np

from sklearn import preprocessing, ensemble, linear_model, model_selection, neural_network, svm, tree
from statistics import mean
from tabulate import tabulate

parser = argparse.ArgumentParser(description='Create models to predict the outcomes of future races')
parser.add_argument('-i', '--iterations', type=int, dest='iterations', help='Number of times to test regressors', default=10)
parser.add_argument('-t', '--training-file', type=str, dest='tfile', help='Data to train models over', required=True)
parser.add_argument('-q', '--quiet', type=bool, const=True, nargs='?', dest='quiet', help='Only print output table and errors', default=False)
parser.add_argument('-v', '--verbose', type=bool, const=True, nargs='?', dest='verbose', help='Print extra information', default=False)

args = parser.parse_args()
iterations = args.iterations
tfile = args.tfile
quiet = args.quiet
verbose = args.verbose

if quiet and verbose:
    print('Quiet and verbose both set to True, defaulting to quiet')

verbose_print = print if verbose and not quiet else lambda *a, **k: None
quiet_print = print if not quiet else lambda *a, **k: None

horse_data = np.load(tfile)
if horse_data.shape[1] != 9:
    raise np.AxisError('Incorrect input size: given', horse_data.shape[1], 'expected 9')

verbose_print('Data loaded from', tfile)

# Clean up data, the row of headers is removed, all rows with an incorrectly
# parsed numeric value are removed, string data is encoded, and features are
# scaled to have mean of 0 and stddev of 1.
horse_data = horse_data[1:, :]
horse_data = horse_data[np.char.isnumeric(horse_data[:, 2])]
horse_data = horse_data[np.char.isnumeric(horse_data[:, 3])]

horse_data[1:, 2:4] = horse_data[1:, 2:4].astype(np.int)

horse_encoder = preprocessing.LabelEncoder()
horse_encoder.fit(horse_data[:, 4])
horse_data[:, 4] = horse_encoder.transform(horse_data[:, 4])

jockey_encoder = preprocessing.LabelEncoder()
jockey_encoder.fit(horse_data[:, 5])
horse_data[:, 5] = jockey_encoder.transform(horse_data[:, 5])

horse_input = horse_data[:,2:6].astype(np.float)
horse_output = horse_data[:, 6:].astype(np.float)

verbose_print('Data cleaned')

for col in range(horse_input.shape[1]):
    scaler = preprocessing.StandardScaler()
    horse_input[:, col] = scaler.fit_transform(horse_input[:,col].reshape(-1,1)).reshape(1, -1)

verbose_print('Data scaled')
    
classifiers = [
    linear_model.PoissonRegressor(warm_start=True),
    linear_model.TweedieRegressor(warm_start=True),
    linear_model.LinearRegression(),
    ensemble.BaggingRegressor(),
    ensemble.ExtraTreesRegressor(),
    ensemble.GradientBoostingRegressor(warm_start=True),
    ensemble.RandomForestRegressor(),
    neural_network.MLPRegressor(hidden_layer_sizes=(32, 16, 8), warm_start=True),
    svm.SVR(),
    tree.DecisionTreeRegressor()
]

verbose_print('Classifiers initialized')

clf_scores = {c.__class__.__name__:[] for c in classifiers}
clf_times = {c.__class__.__name__:[] for c in classifiers}

quiet_print('\n================================================================')
for i in range(iterations):
    quiet_print(('Split #' + str(i + 1)).ljust(30), 'Fit time  ', 'Total time', 'Score     |', sep='|')
    quiet_print('----------------------------------------------------------------')
    x_train, x_test, y_train, y_test = model_selection.train_test_split(horse_input, horse_output[:,2], test_size=0.1, random_state = i)

    for clf in classifiers:
        verbose_print('Started ', clf.__class__.__name__.ljust(30), 'at', time.time())
        start_time = time.time()
        clf.fit(x_train, y_train)
        fit_time = time.time() - start_time
        verbose_print('Trained ', clf.__class__.__name__.ljust(30), 'at', time.time())
        score = clf.score(x_test, y_test)
        end_time = time.time() - start_time
        verbose_print('Finished', clf.__class__.__name__.ljust(30), 'at', time.time())

        clf_scores[clf.__class__.__name__].append(score)
        clf_times[clf.__class__.__name__].append(end_time)
        quiet_print(clf.__class__.__name__.ljust(30), str(round(fit_time, 2)).rjust(10), str(round(end_time, 2)).rjust(10), str(round(score, 5)).rjust(10) + '|', sep='|')
    
    quiet_print('================================================================')


print('\n========================================================================')
print('Final scores'.ljust(30), 'Average Score'.ljust(20), 'Average Time'.ljust(20) + '|', sep='|')
print('------------------------------------------------------------------------')
for clf in clf_scores.keys():
    print(clf.ljust(30), str(round(mean(clf_scores[clf]), 5)).rjust(20), str(round(mean(clf_times[clf]), 5)).rjust(20) + '|', sep='|')
print('========================================================================')