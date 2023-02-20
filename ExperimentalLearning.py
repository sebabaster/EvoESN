import numpy as np
#
# Teacher-forced is applied during ALL training and in the vanishing windows for testing.
# errorType 0 is squared error in H, errorType 1 is NRMSE_H
# by default is using H=100
Learning_config_Lorenz = {'trainInit': 1000, \
                   'trainEnd': 6000, \
                   'testInit': 6000, \
                   'testEnd': 7600,\
                   'teacherForced': 1000, \
                   'learningMode': 0, \
                   'inSize': 1, \
                   'timeAhead': 1, \
                   'errorType': 0, \
                   'errorHorizon': 100, \
                  }
#
Learning_config_MGS = {'trainInit': 1000, \
                    'trainEnd': 3000, \
                    'testInit': 3000, \
                    'testEnd': 5600, \
                    'teacherForced': 1000, \
                    'learningMode': 0, \
                   'inSize': 1, \
                   'timeAhead': 1, \
                   'errorType': 0, \
                   'errorHorizon': 100, \
                   }
#
Learning_config_Hennon = {'trainInit': 200, \
                    'trainEnd': 2500, \
                    'testInit': 2500, \
                    'testEnd': 4790, \
                    'teacherForced': 1000, \
                    'learningMode': 0, \
                   'inSize': 1, \
                   'timeAhead': 1, \
                   'errorType': 0, \
                   'errorHorizon': 3, \
                   }

Learning_config_SNMonth = {'trainInit': 200, \
                    'trainEnd': 2000, \
                    'testInit': 2000, \
                    'testEnd': 3270, \
                    'teacherForced': 0, \
                    'learningMode': 0, \
                   'inSize': 1, \
                   'timeAhead': 1, \
                   'errorType': 0, \
                   'errorHorizon': 10, \
                   }
#
Learning_config_SNDay = {'trainInit': 200, \
                    'trainEnd': 2500, \
                    'testInit': 2500, \
                    'testEnd': 4790, \
                    'teacherForced': 1000, \
                    'learningMode': 0, \
                   'inSize': 1, \
                   'timeAhead': 1, \
                   'errorType': 0, \
                   'errorHorizon': 3, \
                   }

#
ESN_param_Lorenz = {'inSize': 1, \
                    'outSize': 1, \
                    'resSize': 500, \
                    'spectralRadius': 0.97, \
                    'density': 0.2, \
                    'leakyRate': 1, \
                    'regParameter': 1 / 10 ** 6, \
                    'noiseParameter': 1 / 10 ** 8, \
                    'rhoControl': 1, \
                    'constantBias': 0.02, \
                    'BiasReservoirOut': 1, \
                    'stateType': 3, \
                    'fedParameter': 1, \
                    'OutputFunction': 1, \
                    'cteIn': [2,1], \
                    'cteRes': [1,0.5], \
                    'cteFed': [8,4], \
                    }
#
ESN_param_MGS = {'inSize': 1, \
                 'outSize': 1, \
                 'resSize': 1000, \
                 'spectralRadius': 0.8, \
                 'density': 0.1, \
                 'leakyRate': 1, \
                 'regParameter':  1/10**9, \
                 'noiseParameter': 1/10**10, \
                 'rhoControl': 1, \
                 'constantBias': 0.2, \
                 'BiasReservoirOut': 1, \
                 'stateType': 3, \
                 'fedParameter': 1, \
                 'OutputFunction': 1,\
                 'cteIn': [2, 1], \
                 'cteRes': [1, 0.5], \
                 'cteFed': [2, 1], \
                 }
#
ESN_param_Hennon = {'inSize': 1, \
                 'outSize': 1, \
                 'resSize': 250, \
                 'spectralRadius': 0.8, \
                 'density': 0.3, \
                 'leakyRate': 1, \
                 'regParameter':  1/10**6, \
                 'noiseParameter': 1/10**8, \
                 'rhoControl': 1, \
                 'constantBias': 0.2, \
                 'BiasReservoirOut': 1, \
                 'stateType': 3, \
                 'fedParameter': 0, \
                 'OutputFunction': 1,\
                 'cteIn': [2, 1], \
                 'cteRes': [1, 0.5], \
                 'cteFed': [2, 1], \
                 }

#
#ESN_param_Sunspot: ESN_param
#{'inSize': 1, 'outSize': 1, 'resSize': 500, 'spectralRadius': 0.85, 'density': 0.5, 'leakyRate': 0.7, 'regParameter': 1e-05, 'noiseParameter': 0, 'rhoControl': 1, 'constantBias': 0.5, 'BiasReservoirOut': 1, 'stateType': 3, 'fedParameter': 0, 'OutputFunction': 1, 'cteIn': [2, 1], 'cteRes': [1, 0.5], 'cteFed': [2, 1], 'dens': 2.5}
#
#ESN_param_SNMonth = ESN_param_Hennon
#ESN_param_SNDay = ESN_param_Hennon
#ESN_param_SNDay["resSize"]=500
