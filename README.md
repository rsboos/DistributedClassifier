# DistributedClassifier
Final project for bSC Computer Science UFRGS. A distributed multi class classifier.

Execution should be in the format: python main.py dataset.txt classesPlace(first or last) algorithmType ( distributed or local) function (combine function or classify algorithm) classifiersType (optional => normal or ova) kFoldNumber(optional => default = 10) subsetFeatures (optional)

AlgorithmType = {distributed,local}
ClassifyAlgorithm = { SVM, DecisionTree, KNN, NN, NB, ECOC }
SocialChoiceFunction = { Borda, Copeland, Dowdall, Plurality }
