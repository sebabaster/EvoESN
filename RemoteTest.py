# Libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, dct, idct
#from sklearn.metrics import mean_squared_error
import random
import array
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import sys, getopt
################
# Developed functions
################
from ESN import *
from initializationESN import *
from EVOESN import *
from Evaluation import *
from numpy import genfromtxt

#######################
# Read benchmark: data, experimental setting and initial parameters of structures.
# Problem names: "Lorenz", "MGS"
# Returns: data, dataTest, ESN_param, Learning_config, GA_param
def initialSetup(selectedProblem):
    if selectedProblem in ["MGS", "Mackey-Glass", "Mackey Glass"]:
        print("Experimental setting for MGS")
        data = np.load("data/mackey_glass_t17.npy")  # data was already preprocessed as indicated in the paper (normalization, etc.).
        Learning_config = {'trainInit': 1000, \
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
        ESN_param = {'inSize': 1, \
                     'outSize': 1, \
                     'resSize': 1000, \
                     'spectralRadius': 0.8, \
                     'density': 0.1, \
                     'leakyRate': 1, \
                     'regParameter':  1/10**9, \
                     'topologyType': 0, \
                     'noiseParameter': 1/10**10, \
                     'rhoControl': 1, \
                     'constantBias': 0.2, \
                     'BiasReservoirOut': 0, \
                     'stateType': 3, \
                     'fedParameter': 1, \
                     'OutputFunction': 1, \
                     'cteIn': [2, 1] , \
                     'cteFed': [2, 1] , \
                     'cteRes': [1, 0.5], \
                     "errorRangeTop": 84,
                     }
    else:
        # I assume that is selected Lorenz benchmark
        print("Experimental setting for Lorenz")
        data = np.load("data/LorenzScaledXdata.npy") # data was already preprocessed as indicated in the paper.
        Learning_config = {'trainInit': 1000, \
                           'trainEnd': 6000, \
                           'testInit': 6000, \
                           'testEnd': 7600, \
                           'teacherForced': 1000, \
                           'learningMode': 0, \
                           'inSize': 1, \
                           'timeAhead': 1, \
                           'errorType': 0, \
                           'errorHorizon': 100, \
                           }
        ESN_param = {'inSize': 1, \
                     'outSize': 1, \
                     'resSize': 500, \
                     'spectralRadius': 0.97, \
                     'density': 0.2, \
                     'leakyRate': 1, \
                     'regParameter': 1/10**6, \
                     'topologyType': 0, \
                     'noiseParameter': 1 /10**7, \
                     'rhoControl': 1, \
                     'constantBias': 0.02, \
                     'BiasReservoirOut': 0, \
                     'stateType': 3, \
                     'fedParameter': 1, \
                     'OutputFunction': 0, \
                     'cteIn': [2, 1] , \
                     'cteFed': [8, 4] , \
                     'cteRes': [1, 0.5], \
                     "errorRangeTop": 84,
                     }
    GA_param = {'INDIVIDUAL_SIZE': 30, \
                'N_GEN': 10, \
                'MUTPB': 0.1, \
                'CXPB': 0.5, \
                'MU': 0.1, \
                'SIGMA': 0.2, \
                'TOURNSIZE': 5, \
                'POPSIZE': 30, \
                }
    dataTest = data[Learning_config['testInit']:Learning_config['testEnd']]
    return data, dataTest, ESN_param, Learning_config, GA_param

#########################

PSO_param = {'INDIVIDUAL_SIZE': 150, \
                'N_GEN': 50, \
                'C1': 2, \
                'C2': 2, \
                'w': 0.9, \
                'POPSIZE': 20, \
                }



def evalParticle(particle, evalParam, y_true):
    # particle extension
    W=evalParam[3]
    Mask=evalParam[4]
    numberCoeff = W['Wres'][Mask].shape[0]
    dataTest = evalParam[2][evalParam[1]['testInit']:evalParam[1]['testEnd']]
    # extended coefficient vector
    coeff = np.hstack([particle, np.zeros(numberCoeff - len(particle))])
    weights = idct(coeff, norm="ortho")
    # Update weights into the reservoir matrix in specific positions given by Mask:
    W['Wres'][Mask] = weights
    [W, x, y,X] = ESNmodel(evalParam[0], evalParam[1], evalParam[2], W)
    predFR = EvaluateFreeRun(evalParam[0], evalParam[1], dataTest, W, x)
    y_pred = predFR[0, evalParam[1]["teacherForced"]:evalParam[1]["teacherForced"] + evalParam[0]["errorRangeTop"]]
    y_std = np.var(dataTest, ddof=1)  # I made the variance over the whole testing data, it can be different too.
    #errorCoeff = np.sqrt(np.mean((testRes[0][83] - dataTest[84]) ** 2))
    #return np.sqrt(np.square(np.subtract(y_true, y_pred)).mean()) / y_std
    #return np.sqrt(np.square(y_true[83]-y_pred[82])/y_std)
    return np.square(np.subtract(y_true, y_pred)).mean() / y_std
    #return np.sqrt(np.square(np.subtract(y_true, y_pred)).mean())

def PSO(PSO_param, evalParam):
    # start_time = time.time()
    # np.random.seed(100)
    # To save spectra of all for showing
    # evalParam has the following form: [ESN_param, Learning_config, data, W, Mask]
    dataTest = evalParam[2][evalParam[1]['testInit']:evalParam[1]['testEnd']]
    y_true = dataTest[evalParam[1]["teacherForced"] + 1:evalParam[1]["teacherForced"] + evalParam[0]["errorRangeTop"] + 1]
    W = evalParam[3]
    Mask = evalParam[4]
    numberCoeff = W['Wres'][Mask].shape[0]
    #
    c1min = 0.5
    c1max = 2.5
    c2min = 0.5
    c2max = 2.5
    c1 = (c1min - c1max) * (1 / PSO_param['N_GEN']) + c1max
    c2 = (c2max - c2min) * (1 / PSO_param['N_GEN']) + c2max
    # c1 = PSO_param['C1']
    # c2 = PSO_param['C2']
    w = PSO_param['w']
    nroParticles = PSO_param["POPSIZE"]
    particleSize = PSO_param["INDIVIDUAL_SIZE"]
    #
    X = np.zeros(shape=(PSO_param["POPSIZE"], PSO_param["INDIVIDUAL_SIZE"]))
    V = np.zeros(shape=(PSO_param["POPSIZE"], PSO_param["INDIVIDUAL_SIZE"]))
    obj = np.zeros(shape=(PSO_param["POPSIZE"], 1))
    pbest_obj = np.zeros(shape=(PSO_param["POPSIZE"], 1))
    gbest_history = np.zeros(shape=(PSO_param['N_GEN']+1, 1))
    spectra = np.zeros(shape=(PSO_param['N_GEN']+1, 1))
    #
    # Initialization
    for i in range(PSO_param["POPSIZE"]):
        #Wres = np.random.randint(8, 12) * (np.random.rand(evalParam[0]['resSize'], evalParam[0]['resSize']) - 0.5)
        Wres = 10 * (np.random.rand(evalParam[0]['resSize'], evalParam[0]['resSize']) - 0.5)
        shortCoeff = np.zeros(shape=(numberCoeff))
        shortCoeff[0:PSO_param["INDIVIDUAL_SIZE"]] = Wres[Mask][0:PSO_param["INDIVIDUAL_SIZE"]]
        X[i, :] = shortCoeff[0:PSO_param["INDIVIDUAL_SIZE"]]
        #
        pbest_obj[i, 0] = evalParticle(X[i, :], evalParam, y_true)
        gbest_obj = pbest_obj.min()
        gbest_history[0,0] = gbest_obj
        # This vector is initialized but is not being used, for using shall uncomment the lines inside the for
        spectra[0,0] =evalParam[0]["spectralRadius"]

    ####
    # Initialize data
    pbest = X  # pbest has the best position of the particle i during the evolution.
    gbest = pbest[pbest_obj.argmin(), :]
    #  Optional scaling
    # for i in range(PSO_param["POPSIZE"]):
    #    coeff = np.concatenate((X[i,:], np.zeros(numberCoeff - PSO_param["INDIVIDUAL_SIZE"])))
    #    weights = idct(coeff, norm="ortho")
    #    W['Wres'][Mask] = weights
    #    rhoW = max(abs(np.linalg.eig(W['Wres'])[0]))
    #    W['Wres'][Mask] *= ESN_param["spectralRadius"]/rhoW
    #    aux = dct(W["Wres"][Mask], norm="ortho")[0:PSO_param["INDIVIDUAL_SIZE"]]
    #    for j in range(PSO_param["INDIVIDUAL_SIZE"]):
    #        X[i,j]= aux[j]
    ####
    for i in range(PSO_param['N_GEN']):
        # Update params
        r1, r2 = np.random.rand(2)
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
        # V = w * V + c1 (pbest - X) + c2 * (gbest - X)
        X = X + V
        # evaluate population
        for j in range(PSO_param["POPSIZE"]):
            # obj[j, 0] = evaluateFullEvo(X[j, :], evalParam)
            # obj[j, 0] = evaluateInd(X[j, :], evalParam)[0]
            obj[j, 0] = evalParticle(X[j, :], evalParam, y_true)

        pbest[(pbest_obj >= obj)[:, 0], :] = X[(pbest_obj >= obj)[:, 0], :]
        pbest_obj = np.array([pbest_obj, obj]).min(axis=0)
        gbest = pbest[pbest_obj.argmin()]
        ##
        # Optional scaling
        #coeff = np.concatenate((pbest[pbest_obj.argmin()], np.zeros(numberCoeff - PSO_param["INDIVIDUAL_SIZE"])))
        #weights = idct(coeff, norm="ortho")
        #W['Wres'][Mask] = weights
        #spectra[i+1,0] = max(abs(np.linalg.eig(W['Wres'])[0]))
        ####
        gbest_obj = pbest_obj.min()
        gbest_history[i+1,0] = gbest_obj
        # Update PSO parameters
        c1 = (c1min - c1max) * (i / PSO_param['N_GEN']) + c1max
        c2 = (c2max - c2min) * (i / PSO_param['N_GEN']) + c2min

    # t = float(time.time() - start_time)
    # print("--- %s seconds ---" % t)
    return gbest_history, gbest, spectra

# Evaluate k random ESNs
def baselineESN(evalParam, K):
    """
    evalParam is a list containing all the necessary elements for running the ESN.
    evalParam has the form: [ESN_param, Learning_config, data, W, Mask]
    """
    dataTest = evalParam[2][evalParam[1]['testInit']:evalParam[1]['testEnd']]
    #
    y_true = dataTest[evalParam[1]["teacherForced"] + 1:evalParam[1]["teacherForced"] + evalParam[0]["errorRangeTop"] + 1]
    #
    W = evalParam[3] # I save the Win used for PSO.
    #
    rhoVector=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99,1.1,1.2,1.3,1.4,1.5]
    densityVector = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    leakyRate = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    errorCoeff = np.zeros(shape=(K, 1))
    for rho in range(len(rhoVector)):
        evalParam[0]["spectralRadius"] = rhoVector[rho]
        for density in range(len(densityVector)):
            evalParam[0]['density'] = densityVector[density]
            for leaky in range(len(leakyRate)):
                evalParam[0]['leakyRate'] = leakyRate[leaky]
                for k in range(K):
                    print("K:", k, "spectral radius:", evalParam[0]["spectralRadius"])
                    Wnew = NetworkInitializationGeneralGraph(evalParam[0])
                    Wnew["Win"] = W["Win"]
                    [W, x, y, X] = ESNmodel(evalParam[0], evalParam[1], evalParam[2], Wnew)
                    predFR = EvaluateFreeRun(evalParam[0],evalParam[1], dataTest, W, x)
                    #######
                    # Compute the error without counting points in the vanishing windows
                    #######
                    y_pred = predFR[0, evalParam[1]["teacherForced"]:evalParam[1]["teacherForced"]+evalParam[0]["errorRangeTop"]]
                    y_std = np.var(dataTest, ddof=1) # I made the variance over the whole testing data, it can be different too.
                    errorCoeff[k, 0] = np.sqrt(np.square(np.subtract(y_true, y_pred)).mean() / y_std)
                    #errorCoeff = evaluation("NRMSE", y_true, y_pred, y_std)
                    #errorCoeff = np.sqrt(mean_squared_error(y_true, y_pred))/y_std
                    #errorCoeff = ESNminimalScript(W, evalParam[0], evalParam[1], evalParam[2])[0]
                    # it return the first element in a tuple (x, )
                    #errores[k,0] = errorCoeff
                    np.save("Proj-"+str(k)+"-"+str(evalParam[0]["resSize"])+"-"+str(evalParam[0]["spectralRadius"])+"-"+str(evalParam[0]["density"])+"-"+str(evalParam[0]["leakyRate"]), X)
                    np.save("Pred-"+str(k)+"-"+str(evalParam[0]["resSize"])+"-"+str(evalParam[0]["spectralRadius"])+"-"+str(evalParam[0]["density"])+"-"+str(evalParam[0]["leakyRate"]), y_pred)
                    np.save("Error-"+str(k)+"-"+str(evalParam[0]["resSize"])+"-"+str(evalParam[0]["spectralRadius"])+"-"+str(evalParam[0]["density"])+"-"+str(evalParam[0]["leakyRate"]), errorCoeff)

    print("Finalization.")


def main():
    ######
    # Initial ESN
    #data, dataTest, ESN_param, Learning_config, GA_param = initialSetup("MGS")
    data, dataTest, ESN_param, Learning_config, GA_param = initialSetup("Lorenz")
    ESN_param["BiasReservoirOut"] = 0
    # W = NetworkInitializationScaleFree(ESN_param)
    W = NetworkInitializationGeneralGraph(ESN_param)
    Mask = W['Wres'] != 0
    #Learning_config["testEnd"] = 5100
    evalParam = [ESN_param, Learning_config, data, W, Mask]
    #K=500
    #print("Here1")
    #errores = baselineESN(evalParam, K)
    #print("Here")
    numberCoeff = W['Wres'][Mask].shape[0]
    # to check different inertia
    K=30
    #baselineESN(evalParam, K)
    # Initial setup
    #data, dataTest, ESN_param, Learning_config, GA_param = initialSetup("Lorenz")
    # print("Data:", data.shape)
    # print("Data:", data.shape)
    # to check different inertia
    #Learning_config["testEnd"]=5100 # because I evaluating with 84 points after teaching force.
    #iter=3
    PSO_param = {'INDIVIDUAL_SIZE': 50, \
                 'N_GEN': 150, \
                 'C1': 0.5, \
                 'C2': 0.5, \
                 'w': 0.9, \
                 'POPSIZE': 30, \
                 }
    #W = NetworkInitializationGeneralGraph(ESN_param)
    #Mask = W['Wres'] != 0
    #evalParam = [ESN_param, Learning_config, data, W, Mask]
    #numberCoeff = W['Wres'][Mask].shape[0]
    #indSize=[750,500,300,150,100,50]
    #indSize = [50]
    #gbest_history=np.zeros(shape=(len(indSize),iter,PSO_param["N_GEN"]))
    #ind=0
    #t1=0
    #PSO_param["INDIVIDUAL_SIZE"] = indSize[ind]
    #aux0, pbest, spectra = PSO(PSO_param, evalParam)
    # Initial ESN
    W = NetworkInitializationGeneralGraph(ESN_param)
    Mask = W['Wres'] != 0
    evalParam = [ESN_param, Learning_config, data, W, Mask]
    numberCoeff = W['Wres'][Mask].shape[0]
    aux0, pbest, spectra = PSO(PSO_param, evalParam)
    np.save('gbest_historyExperiment.npy', aux0)
    np.save('spectra_historyExperiment.npy', spectra)

if __name__ == "__main__":
    main()