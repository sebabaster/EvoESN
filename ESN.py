import numpy as np
import matplotlib.pyplot as plt
#from sklearn.metrics import mean_squared_error
import random
import initializationESN

########################################
def computeState(ESN_param,W,input,previousState,previousPrediction):
    # previousPrediction is used only when we have feedback connections
    x = np.random.rand(ESN_param['resSize'], 1)
    if ESN_param['stateType'] == 0:  # x= (1-delta)x + delta*(htan((Win u) + Wx))
        x = (1 - ESN_param['leakyRate']) * previousState + ESN_param['leakyRate'] * np.tanh(np.dot(W['Win'], np.vstack((ESN_param['constantBias'], input))) + np.dot(W['Wres'], previousState))
    elif ESN_param['stateType'] == 1:  # x= (1-delta)x + htan(Win u + delta(Wres x))
        x = (1 - ESN_param['leakyRate']) * previousState + np.tanh(np.dot(W['Win'], np.vstack((ESN_param['constantBias'], input))) + ESN_param['leakyRate'] * np.dot(W['Wres'], previousState))
    elif ESN_param['stateType'] == 2:  # x = (1-delta)*x + htan(Win u + delta * Wres x + noise)
        aux = ESN_param['leakyRate'] * np.dot(W['Wres'], previousState)[:, 0] + ESN_param['noiseParameter'] * (np.random.rand(1, ESN_param['resSize']) - 0.5)
        x = (1 - ESN_param['leakyRate']) * previousState + np.tanh(np.dot(W['Win'], np.vstack((ESN_param['constantBias'], input))) + aux.transpose())
    else:
        x = np.tanh(np.dot(W['Win'], np.vstack((ESN_param['constantBias'], input))) + np.dot(W['Wres'], previousState) + ESN_param['fedParameter'] * np.dot(W['Wfed'], previousPrediction) + ESN_param['noiseParameter'] * (np.random.rand(ESN_param['resSize'], 1) - 0.5))

    return x

def computeOutput(ESN_param,W,input,state):
    if ESN_param["OutputFunction"] == 0:
        if ESN_param["BiasReservoirOut"] == 1:
            y = np.dot(W['Wout'].transpose(), np.vstack((1, input, state)))
        else:
            y = np.dot(W['Wout'].transpose(), np.vstack((input, state)))
    else:
        if ESN_param["BiasReservoirOut"] == 1:
            y = np.tanh(np.dot(W['Wout'].transpose(), np.vstack((1, input, state))))
        else:
            y = np.tanh(np.dot(W['Wout'].transpose(), np.vstack((input, state))))
    return y

def computeWout(ESN_param, Learning_config, data, X):
    Yt = data[None, Learning_config['timeAhead']:(data.shape[0])]  # shift one time-step.
    Xrange = X.transpose()
    if ESN_param["OutputFunction"] == 0:
        if ESN_param["BiasReservoirOut"] == 1:
            Wout = np.dot(np.dot(Yt[0, Learning_config['trainInit']:Learning_config['trainEnd']],Xrange[Learning_config['trainInit']:Learning_config['trainEnd'], :]),np.linalg.inv(np.dot(X[:, Learning_config['trainInit']:Learning_config['trainEnd']],Xrange[Learning_config['trainInit']:Learning_config['trainEnd'],:]) + ESN_param['regParameter'] * np.identity(1 + ESN_param['inSize'] + ESN_param['resSize'])))
        else:
            Wout = np.dot(np.dot(Yt[0, Learning_config['trainInit']:Learning_config['trainEnd']],Xrange[Learning_config['trainInit']:Learning_config['trainEnd'], :]),np.linalg.inv(np.dot(X[:, Learning_config['trainInit']:Learning_config['trainEnd']],Xrange[Learning_config['trainInit']:Learning_config['trainEnd'],:]) + ESN_param['regParameter'] * np.identity(ESN_param['inSize'] + ESN_param['resSize'])))

        y = np.dot(Xrange, Wout)
    else:
        if ESN_param["BiasReservoirOut"] == 1:
            Wout = np.dot(np.dot(np.arctanh(Yt[0, Learning_config['trainInit']:Learning_config['trainEnd']]), Xrange[Learning_config['trainInit']:Learning_config['trainEnd'], :]),np.linalg.inv(np.dot(X[:, Learning_config['trainInit']:Learning_config['trainEnd']],Xrange[Learning_config['trainInit']:Learning_config['trainEnd'],:]) + ESN_param['regParameter'] * np.identity(1 + ESN_param['inSize'] + ESN_param['resSize'])))
        else:
            Wout = np.dot(np.dot(np.arctanh(Yt[0, Learning_config['trainInit']:Learning_config['trainEnd']]),Xrange[Learning_config['trainInit']:Learning_config['trainEnd'], :]),np.linalg.inv(np.dot(X[:, Learning_config['trainInit']:Learning_config['trainEnd']],Xrange[Learning_config['trainInit']:Learning_config['trainEnd'],:]) + ESN_param['regParameter'] * np.identity(ESN_param['inSize'] + ESN_param['resSize'])))

        y = np.tanh(np.dot(Xrange, Wout))

    return y, Wout

def ESNmodel(ESN_param, Learning_config, data, W):
        """
        Return [W, x, y],
        """
        #Yt = data[None, Learning_config['timeAhead']:(data.shape[0])]  # shift one time-step.
        X = np.zeros(shape=(ESN_param['inSize'] + ESN_param['resSize'], Learning_config['trainEnd']))
        # x state.
        #x = np.random.rand(ESN_param['resSize'], 1)
        x = np.zeros(shape=(ESN_param['resSize'], 1))
        for t in range(Learning_config['trainEnd']):
            u = data[t]  # input pattern
            yOut = data[t]  # in training phase is used the target instead the predicted value.
            x = computeState(ESN_param, W, u, x, yOut)
            X[:, t] = np.vstack((u, x))[:, 0]

        # add the bias term for each sample
        if ESN_param["BiasReservoirOut"] == 1:
            X = np.vstack((np.zeros(shape=(1, Learning_config['trainEnd'])) + 1, X))

        y, W['Wout'] = computeWout(ESN_param, Learning_config, data, X)
        # return weights and last reservoir state.
        return [W, x, y, X]

def EvaluateFreeRun(ESN_param, Learning_config, dataTest, W, x):
    ##
    # to test over data in range test
    # run the trained ESN in a generative mode.
    # it use the x initial state given by parameter.
    # Ytest = data[None, Learning_config['trainEnd']:(Learning_config['trainEnd'] + Learning_config['testLen'])]
    # Ytest = data[None, (Learning_config['initialTest'] + Learning_config['timeAhead']):((Learning_config['initialTest'] + Learning_config['testLen'] + Learning_config['timeAhead']))]
    testLen = Learning_config['testEnd']-Learning_config['testInit']
    pred = np.zeros(shape=(1, testLen))
    #
    u = dataTest[0]
    yOut = dataTest[0]
    ESN_param['noiseParameter'] = 0 # the noise is used only on training phase.

    for t in range(testLen):
        x = computeState(ESN_param, W, u, x, u)
        y = computeOutput(ESN_param, W, u, x)
        if t < Learning_config['teacherForced']:
            u = dataTest[t]
        else:
            u = y[0]

        pred[0][t] = u

    return pred

def EvaluateOneAhead(ESN_param, Learning_config, dataTest, W, x):
    ##
    testLen = Learning_config['testEnd']-Learning_config['testInit']
    pred = np.zeros(shape=(1, testLen))
    u = dataTest[0]
    ESN_param['noiseParameter'] = 0  # the noise is used only on training phase.
    x = computeState(ESN_param, W, u, x, u)
    for t in range(testLen-1):
        x = computeState(ESN_param, W, u, x, u)
        pred[0][t] = computeOutput(ESN_param, W, u, x)
        u = dataTest[t]

    return pred


#####################################
# It is for the refined schema of Jaegger.
#####################################
def secondStageGeneration(ESN_param,Learning_config, W, data, pred):
    # This function implements the ida of refined learning method introduced in Jaeger et al. .....Science, 2004.
    # We apply teacher-forced, the output yOut(t) used as feedback to reservoir is forced to be the target at y(t).
    dataGen=np.zeros(shape=data.shape)
    dataGen[0]=data[0]
    pred[0:Learning_config['trainInit']]=data[1:Learning_config['trainInit']+1]
    x = np.random.rand(ESN_param['resSize'], 1)
    for t in list(range(Learning_config['trainEnd']-1)):
        print("t", t)
        x = np.tanh(np.dot(W['Wres'], x) + ESN_param['fedParameter'] * np.dot(W['Wfed'], pred[t]))
        if t<Learning_config['trainInit']:
            dataGen[t]=data[t]
        else:
            #dataGen[t] = np.tanh(np.dot(x[:,0], W['Wout'][1:]))
            dataGen[t] = np.tanh(np.dot(W['Wout'][1:], x))[0]
    return dataGen
#####################################
# Evaluation functions (old version only for feedback).
#####################################
def ESNmodelV0(ESN_param, Learning_config, data, W):
    """
    Return [W, x, y],
    # The state is computed since sample 0, but it is used for training since trainInit
    """
    Yt = data[None, Learning_config['timeAhead']:(data.shape[0])]  # shift one time-step.
    # x state.
    x = np.random.rand(ESN_param['resSize'], 1)
    X = np.zeros(shape=(ESN_param['inSize'] + ESN_param['resSize'], Learning_config['trainEnd']))
    if ESN_param['stateType'] == 0:  # x= (1-delta)x + delta*(htan((Win u) + Wx))
        for t in range(Learning_config['trainEnd']):
            u = data[t]  # input pattern
            x = (1 - ESN_param['leakyRate']) * x + ESN_param['leakyRate'] * np.tanh(np.dot(W['Win'], np.vstack((ESN_param['constantBias'], u))) + np.dot(W['Wres'], x))
            X[:, t] = np.vstack((u, x))[:, 0]
    elif ESN_param['stateType'] == 1:  # x= (1-delta)x + htan(Win u + delta(Wres x))
        for t in range(Learning_config['trainEnd']):
            u = data[t]  # input pattern
            x = (1 - ESN_param['leakyRate']) * x + np.tanh(np.dot(W['Win'], np.vstack((ESN_param['constantBias'], u))) + ESN_param['leakyRate'] * np.dot(W['Wres'], x))
            X[:, t] = np.vstack((u, x))[:, 0]
    elif ESN_param['stateType'] == 2:  # x = (1-delta)*x + htan(Win u + delta * Wres x + noise)
        for t in range(Learning_config['trainEnd']):
            u = data[t]  # input pattern
            aux = ESN_param['leakyRate'] * np.dot(W['Wres'], x)[:, 0] + ESN_param['noiseParameter'] * (np.random.rand(1, ESN_param['resSize']) - 0.5)
            x = (1 - ESN_param['leakyRate']) * x + np.tanh(np.dot(W['Win'], np.vstack((ESN_param['constantBias'], u))) + aux.transpose())
            X[:, t] = np.vstack((u, x))[:, 0]
    else:
        # Y = np.zeros(shape=(1,testLen))
        for t in range(Learning_config['trainEnd']):
            u = data[t]  # input pattern
            yOut = data[t]  # when t is 0 it is assigned a random value, in that case the last one in the data series.
            # compute state.
            # We apply teacher-forced, the output yOut(t) used as feedback to reservoir is forced to be the target at y(t).
            x = np.tanh(np.dot(W['Win'], np.vstack((ESN_param['constantBias'], u))) + np.dot(W['Wres'], x) + ESN_param[
                'fedParameter'] * np.dot(W['Wfed'], yOut) + ESN_param['noiseParameter'] * (
                                    np.random.rand(ESN_param['resSize'], 1) - 0.5))
            X[:, t] = np.vstack((u, x))[:, 0]

    if ESN_param["BiasReservoirOut"] == 1:
        # add the bias term for each sample
        X = np.vstack((np.zeros(shape=(1,Learning_config['trainEnd']))+1,X))

    # Compute Wout
    Xrange = X.transpose()
    if ESN_param["OutputFunction"] == 0:
        if ESN_param["BiasReservoirOut"] == 1:
            W['Wout'] = np.dot(np.dot(Yt[0, Learning_config['trainInit']:Learning_config['trainEnd']], Xrange[Learning_config['trainInit']:Learning_config['trainEnd'], :]), np.linalg.inv(np.dot(X[:, Learning_config['trainInit']:Learning_config['trainEnd']], Xrange[Learning_config['trainInit']:Learning_config['trainEnd'], :]) + ESN_param['regParameter'] * np.identity(1 + ESN_param['inSize'] + ESN_param['resSize'])))
        else:
            W['Wout'] = np.dot(np.dot(Yt[0, Learning_config['trainInit']:Learning_config['trainEnd']],Xrange[Learning_config['trainInit']:Learning_config['trainEnd'], :]), np.linalg.inv(np.dot(X[:, Learning_config['trainInit']:Learning_config['trainEnd']],Xrange[Learning_config['trainInit']:Learning_config['trainEnd'], :]) + ESN_param['regParameter'] * np.identity(ESN_param['inSize'] + ESN_param['resSize'])))

        y = np.dot(Xrange, W['Wout'])
    else:
        if ESN_param["BiasReservoirOut"] == 1:
            W['Wout'] = np.dot(np.dot(np.arctanh(Yt[0, Learning_config['trainInit']:Learning_config['trainEnd']]),Xrange[Learning_config['trainInit']:Learning_config['trainEnd'], :]),np.linalg.inv(np.dot(X[:, Learning_config['trainInit']:Learning_config['trainEnd']],Xrange[Learning_config['trainInit']:Learning_config['trainEnd'], :]) + ESN_param['regParameter'] * np.identity(1 + ESN_param['inSize'] + ESN_param['resSize'])))
        else:
            W['Wout'] = np.dot(np.dot(np.arctanh(Yt[0, Learning_config['trainInit']:Learning_config['trainEnd']]),Xrange[Learning_config['trainInit']:Learning_config['trainEnd'], :]),np.linalg.inv(np.dot(X[:, Learning_config['trainInit']:Learning_config['trainEnd']],Xrange[Learning_config['trainInit']:Learning_config['trainEnd'],:]) + ESN_param['regParameter'] * np.identity(ESN_param['inSize'] + ESN_param['resSize'])))

        y = np.tanh(np.dot(Xrange, W['Wout']))

    # return weights and last reservoir state.
    return [W, x, y]
#
def EvaluateFreeRunV0(ESN_param, Learning_config, dataTest, W, x):
    ##
    # to test over data in range test
    # run the trained ESN in a generative mode.
    # it use the x initial state given by parameter.
    # Ytest = data[None, Learning_config['trainEnd']:(Learning_config['trainEnd'] + Learning_config['testLen'])]
    # Ytest = data[None, (Learning_config['initialTest'] + Learning_config['timeAhead']):((Learning_config['initialTest'] + Learning_config['testLen'] + Learning_config['timeAhead']))]
    testLen = Learning_config['testEnd']-Learning_config['testInit']
    pred = np.zeros(shape=(1, testLen))
    #
    u = dataTest[0]
    ESN_param['noiseParameter'] = 0 # the noise is used only on training phase.
    if ESN_param['stateType'] == 0:  # x= (1-delta)x + delta*(htan((Win u) + Wx))
        for t in range(testLen):
            x = (1 - ESN_param['leakyRate']) * x + ESN_param['leakyRate'] * np.tanh(np.dot(W['Win'], np.vstack((ESN_param['constantBias'], u))) + np.dot(W['Wres'], x))
            if ESN_param["OutputFunction"] == 0:
                if ESN_param["BiasReservoirOut"] == 1:
                    y = np.dot(W['Wout'].transpose(), np.vstack((1, u, x)))
                else:
                    y = np.dot(W['Wout'].transpose(), np.vstack((u, x)))
            else:
                if ESN_param["BiasReservoirOut"] == 1:
                    y = np.tanh(np.dot(W['Wout'].transpose(), np.vstack((1, u, x))))
                    # print("Here1: y:", y[0])
                else:
                    y = np.tanh(np.dot(W['Wout'].transpose(), np.vstack((u, x))))
                    # print("Here2: y:",y[0])

            if t < Learning_config['teacherForced']:
                print("Here state 0:", t, "u: ", dataTest[t])
                u = dataTest[t]
            else:
                u = y[0]
            pred[0][t] = u
    elif ESN_param['stateType'] == 1:
        for t in range(testLen):
            x = (1 - ESN_param['leakyRate']) * x + np.tanh(np.dot(W['Win'], np.vstack((ESN_param['constantBias'], u))) + ESN_param['leakyRate'] * np.dot(W['Wres'],x))
            if ESN_param["OutputFunction"] == 0:
                if ESN_param["BiasReservoirOut"] == 1:
                    y = np.dot(W['Wout'].transpose(), np.vstack((1, u, x)))
                else:
                    y = np.dot(W['Wout'].transpose(), np.vstack((u, x)))
            else:
                if ESN_param["BiasReservoirOut"] == 1:
                    y = np.tanh(np.dot(W['Wout'].transpose(), np.vstack((1, u, x))))
                    # print("Here1: y:", y[0])
                else:
                    y = np.tanh(np.dot(W['Wout'].transpose(), np.vstack((u, x))))
                    # print("Here2: y:",y[0])

            if t < Learning_config['teacherForced']:
                print("Here state 1:", t, "u: ", dataTest[t])
                u = dataTest[t]
            else:
                u = y[0]
            pred[0][t] = u
    elif ESN_param['stateType'] == 2:
        for t in range(testLen):
            aux = ESN_param['leakyRate'] * np.dot(W['Wres'], x)[:, 0] + ESN_param['noiseParameter'] * (
                        np.random.rand(1, ESN_param['resSize']) - 0.5)
            x = (1 - ESN_param['leakyRate']) * x + np.tanh(
                np.dot(W['Win'], np.vstack((ESN_param['constantBias'], u))) + aux.transpose())
            if ESN_param["OutputFunction"] == 0:
                if ESN_param["BiasReservoirOut"] == 1:
                    y = np.dot(W['Wout'].transpose(), np.vstack((1, u, x)))
                else:
                    y = np.dot(W['Wout'].transpose(), np.vstack((u, x)))
            else:
                if ESN_param["BiasReservoirOut"] == 1:
                    y = np.tanh(np.dot(W['Wout'].transpose(), np.vstack((1, u, x))))
                    # print("Here1: y:", y[0])
                else:
                    y = np.tanh(np.dot(W['Wout'].transpose(), np.vstack((u, x))))
                    # print("Here2: y:",y[0])

            if t < Learning_config['teacherForced']:
                print("Here state 2:", t, "u: ", dataTest[t])
                u = dataTest[t]
            else:
                u = y[0]
            pred[0][t] = u
    else:
        pred[0][0] = dataTest[0]  # it can be random as well, because requires a vanishing window.
        pred[0][-1] = dataTest[0]  # it can be random as well, because requires a vanishing window.
        for t in range(testLen):
            #print("t:",t, " - dataTest: ", dataTest[t], " x: ", x[0,0], "- y: ", pred[0][t-1])
            # Made just for state 3.
            yOut = pred[0][t-1]
            x = np.tanh(np.dot(W['Win'], np.vstack((ESN_param['constantBias'], u))) + np.dot(W['Wres'], x) + ESN_param['fedParameter']*np.dot(W['Wfed'], yOut) + ESN_param['noiseParameter'] * (np.random.rand(ESN_param['resSize'], 1)-0.5))
            if ESN_param["OutputFunction"] == 0:
                if ESN_param["BiasReservoirOut"] == 1:
                    y = np.dot(W['Wout'].transpose(), np.vstack((1, u, x)))
                else:
                    y = np.dot(W['Wout'].transpose(), np.vstack((u, x)))
            else:
                if ESN_param["BiasReservoirOut"] == 1:
                    y = np.tanh(np.dot(W['Wout'].transpose(), np.vstack((1, u, x))))
                    # print("Here1: y:", y[0])
                else:
                    y = np.tanh(np.dot(W['Wout'].transpose(), np.vstack((u, x))))
                    # print("Here2: y:",y[0])

            if t < Learning_config['teacherForced']:
                #print("Here state 3 - menor teacherForced:", t, "u: ", dataTest[t])
                u = dataTest[t]
            else:
                #print("Here state 3 - MAJOR teacherForced:", t, "u: ", dataTest[t])
                u = y[0]

            pred[0][t] = u

    return pred
#
def EvaluateFreeRunOLD(ESN_param, Learning_config, dataTest, W, x):
    ##
    # to test over data in range test
    # run the trained ESN in a generative mode.
    # it use the x initial state given by parameter.
    # Ytest = data[None, Learning_config['trainEnd']:(Learning_config['trainEnd'] + Learning_config['testLen'])]
    # Ytest = data[None, (Learning_config['initialTest'] + Learning_config['timeAhead']):((Learning_config['initialTest'] + Learning_config['testLen'] + Learning_config['timeAhead']))]
    testLen = Learning_config['testEnd']-Learning_config['testInit']
    pred = np.zeros(shape=(1, testLen))
    u = dataTest[0]
    pred[0][0] =dataTest[0]  # it can be random as well, because requires a vanishing window.
    pred[0][-1] =dataTest[0]  # it can be random as well, because requires a vanishing window.
    pred[0][-1] = dataTest[0]  # it can be random as well, because requires a vanishing window.
    #
    ESN_param['noiseParameter'] = 0 # the noise is used only on training phase.
    for t in range(testLen):
        #print("t:",t, " - dataTest: ", dataTest[t], " x: ", x[0,0], "- y: ", pred[0][t-1])
        # Made just for state 3.
        yOut = pred[0][t-1]
        x = np.tanh(np.dot(W['Win'], np.vstack((ESN_param['constantBias'], u))) + np.dot(W['Wres'], x) + ESN_param['fedParameter']*np.dot(W['Wfed'], yOut) + ESN_param['noiseParameter'] * (np.random.rand(ESN_param['resSize'], 1)-0.5))
        if ESN_param["OutputFunction"] == 0:
            if ESN_param["BiasReservoirOut"] == 1:
                y = np.dot(W['Wout'].transpose(), np.vstack((1, u, x)))
            else:
                y = np.dot(W['Wout'].transpose(), np.vstack((u, x)))
        else:
            if ESN_param["BiasReservoirOut"] == 1:
                y = np.tanh(np.dot(W['Wout'].transpose(), np.vstack((1, u, x))))
                #print("Here1: y:", y[0])
            else:
                y = np.tanh(np.dot(W['Wout'].transpose(), np.vstack((u, x))))
                #print("Here2: y:",y[0])

        if t < Learning_config['teacherForced']:
            print("Here:", t, "u: ", dataTest[t])
            u = dataTest[t]
        else:
            u = y[0]
        pred[0][t] = u

    return pred
#
# evalParam = [ESN_param, Learning_config, data, W, Mask]
# predFR=EvaluateOneAhead(evalParam[0], evalParam[1], dataTest, W, x)
"""
def EvaluateOneAhead(ESN_param, Learning_confing, dataTest, W, x):
    ##
    # to test over data in range test
    # run the trained ESN in a generative mode.
    # it use the x initial state given by parameter.
    # Ytest = data[None, Learning_config['trainEnd']:(Learning_config['trainEnd'] + Learning_config['testLen'])]
    # Ytest = data[None, (Learning_config['initialTest'] + Learning_config['timeAhead']):((Learning_config['initialTest'] + Learning_config['testLen'] + Learning_config['timeAhead']))]
    testLen = Learning_config['testEnd']-Learning_config['testInit']
    print("TestLen: ", testLen)
    pred = np.zeros(shape=(1, testLen))
    #
    u = dataTest[0]
    ESN_param['noiseParameter'] = 0 # the noise is used only on training phase.
    if ESN_param['stateType'] == 0:  # x= (1-delta)x + delta*(htan((Win u) + Wx))
        for t in range(testLen):
            x = (1 - ESN_param['leakyRate']) * x + ESN_param['leakyRate'] * np.tanh(np.dot(W['Win'], np.vstack((ESN_param['constantBias'], u))) + np.dot(W['Wres'], x))
            if ESN_param["OutputFunction"] == 0:
                if ESN_param["BiasReservoirOut"] == 1:
                    y = np.dot(W['Wout'].transpose(), np.vstack((1, u, x)))
                else:
                    y = np.dot(W['Wout'].transpose(), np.vstack((u, x)))
            else:
                if ESN_param["BiasReservoirOut"] == 1:
                    y = np.tanh(np.dot(W['Wout'].transpose(), np.vstack((1, u, x))))
                    # print("Here1: y:", y[0])
                else:
                    y = np.tanh(np.dot(W['Wout'].transpose(), np.vstack((u, x))))
                    # print("Here2: y:",y[0])
            u = dataTest[t]
            pred[0][t] = y[0]

    elif ESN_param['stateType'] == 1:
        for t in range(testLen):
            x = (1 - ESN_param['leakyRate']) * x + np.tanh(np.dot(W['Win'], np.vstack((ESN_param['constantBias'], u))) + ESN_param['leakyRate'] * np.dot(W['Wres'],x))
            if ESN_param["OutputFunction"] == 0:
                if ESN_param["BiasReservoirOut"] == 1:
                    y = np.dot(W['Wout'].transpose(), np.vstack((1, u, x)))
                else:
                    y = np.dot(W['Wout'].transpose(), np.vstack((u, x)))
            else:
                if ESN_param["BiasReservoirOut"] == 1:
                    y = np.tanh(np.dot(W['Wout'].transpose(), np.vstack((1, u, x))))
                    # print("Here1: y:", y[0])
                else:
                    y = np.tanh(np.dot(W['Wout'].transpose(), np.vstack((u, x))))

            u = dataTest[t]
            pred[0][t] = y[0]

    elif ESN_param['stateType'] == 2:
        for t in range(testLen):
            aux = ESN_param['leakyRate'] * np.dot(W['Wres'], x)[:, 0] + ESN_param['noiseParameter'] * (
                        np.random.rand(1, ESN_param['resSize']) - 0.5)
            x = (1 - ESN_param['leakyRate']) * x + np.tanh(
                np.dot(W['Win'], np.vstack((ESN_param['constantBias'], u))) + aux.transpose())
            if ESN_param["OutputFunction"] == 0:
                if ESN_param["BiasReservoirOut"] == 1:
                    y = np.dot(W['Wout'].transpose(), np.vstack((1, u, x)))
                else:
                    y = np.dot(W['Wout'].transpose(), np.vstack((u, x)))
            else:
                if ESN_param["BiasReservoirOut"] == 1:
                    y = np.tanh(np.dot(W['Wout'].transpose(), np.vstack((1, u, x))))
                    # print("Here1: y:", y[0])
                else:
                    y = np.tanh(np.dot(W['Wout'].transpose(), np.vstack((u, x))))
                    # print("Here2: y:",y[0])
            u = dataTest[t]
            pred[0][t] = y[0]

    else:
        pred[0][0] = dataTest[0]  # it can be random as well, because requires a vanishing window.
        pred[0][-1] = dataTest[0]  # it can be random as well, because requires a vanishing window.
        for t in range(testLen):
            #print("t:",t, " - dataTest: ", dataTest[t], " x: ", x[0,0], "- y: ", pred[0][t-1])
            # Made just for state 3.
            yOut = pred[0][t-1]
            x = np.tanh(np.dot(W['Win'], np.vstack((ESN_param['constantBias'], u))) + np.dot(W['Wres'], x) + ESN_param['fedParameter']*np.dot(W['Wfed'], yOut) + ESN_param['noiseParameter'] * (np.random.rand(ESN_param['resSize'], 1)-0.5))
            if ESN_param["OutputFunction"] == 0:
                if ESN_param["BiasReservoirOut"] == 1:
                    y = np.dot(W['Wout'].transpose(), np.vstack((1, u, x)))
                else:
                    y = np.dot(W['Wout'].transpose(), np.vstack((u, x)))
            else:
                if ESN_param["BiasReservoirOut"] == 1:
                    y = np.tanh(np.dot(W['Wout'].transpose(), np.vstack((1, u, x))))
                    # print("Here1: y:", y[0])
                else:
                    y = np.tanh(np.dot(W['Wout'].transpose(), np.vstack((u, x))))
                    # print("Here2: y:",y[0])
            u = dataTest[t]
            pred[0][t] = y[0]
    #return [W, x, y]
    return [x, pred]


"""