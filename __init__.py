# __init__.py
from .LoadData import dataLorenz
from .LoadData import dataMGS
#
from .ExperimentalLearning import Learning_config_Lorenz
from .ExperimentalLearning import Learning_config_MGS
#
from .ExperimentalLearning import ESN_param_Lorenz
from .ExperimentalLearning import ESN_param_MGS
#
from ESN import *
from initializationESN import *
from EVOESN import *
from Evaluation import *
