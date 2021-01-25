from .lica_learner import LICALearner,LICALearner_CNN

REGISTRY = {}

REGISTRY["lica_learner"] = LICALearner
REGISTRY["lica_learner_cnn"] = LICALearner_CNN
