import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, dct, idct
#from sklearn.metrics import mean_squared_error
import random

###
# I give number of non-zero up diagonals
def NetworkInitializationNumberPermutations(ESN_param, C):
    Win = ESN_param["cteIn"][0] * np.random.rand(ESN_param['resSize'], 1 + ESN_param['inSize']) - ESN_param["cteIn"][1]
    Wfed = ESN_param["cteFed"][0] * np.random.rand(ESN_param['resSize'], ESN_param['outSize']) - ESN_param["cteFed"][1]
    # Like Claudio suggested: identity -> 1 element outside -> permutation.
    Wres = ESN_param["cteFed"][0] * np.random.rand(ESN_param['resSize']) * np.identity(ESN_param['resSize'])

    for j in range(C):
        lower = np.random.rand(ESN_param['resSize']-(j+1))
        np.fill_diagonal(Wres[(j+1):, :-(j+1)], lower)

    Wres = np.random.permutation(Wres)  # makes permutation among rows.
    for i in range(ESN_param['resSize']): # correct the diagonal to avoid loops.
        Wres[i, i] = 0

    if ESN_param["BiasReservoirOut"] == 1:
        Wout = np.zeros(shape=(ESN_param['resSize'] + 2, ESN_param['outSize']))
    else:
        Wout = np.zeros(shape=(ESN_param['resSize'] + 1, ESN_param['outSize']))

    # Spectral radius control
    if ESN_param['rhoControl'] == 1:
        rhoW = max(abs(np.linalg.eig(Wres)[0]))
        Wres *= ESN_param['spectralRadius'] / rhoW

    W = {'Win': Win, 'Wres': Wres, 'Wout': Wout, 'Wfed': Wfed}

    return W

##################
# (DONE)
def NetworkInitializationGeneralGraph(ESN_param):
    """
    Reservoir is initialized with a recurrent graph, it has the parameters for controlling spectral radius and density
    NOTE: Doesn't control existe of recurrences
    Inputs: ESN_param is a dictionary  with all the parameters for the ESN model.
    Returns: Win (input-reservoir weights) and W (reservoir weights), Wfb (feedback connections).
    """
    Win = ESN_param["cteIn"][0]*np.random.rand(ESN_param['resSize'], 1 + ESN_param['inSize']) - ESN_param["cteIn"][1]
    Wfed = ESN_param["cteFed"][0]*np.random.rand(ESN_param['resSize'], ESN_param['outSize']) - ESN_param["cteFed"][1]
    Wres = ESN_param["cteRes"][0] * np.random.rand(ESN_param['resSize'], ESN_param['resSize']) - ESN_param["cteRes"][1]

    if ESN_param["BiasReservoirOut"] == 1:
        Wout = np.zeros(shape=(ESN_param['resSize'] + 2, ESN_param['outSize']))
    else:
        Wout = np.zeros(shape=(ESN_param['resSize'] + 1, ESN_param['outSize']))

    # Density control
    dens = np.random.rand(ESN_param['resSize'], ESN_param['resSize'])
    if ESN_param['density'] < 1:  # density is 0.2 then 80% of elements are zeros
        Wres[np.where(dens > ESN_param['density'])] = 0
    # And anyway it is guaranteed at least one circle.
    lowDiag = ESN_param["cteRes"][0] * np.random.rand(ESN_param['resSize'] - 1)
    np.fill_diagonal(Wres[1:, :-1], lowDiag[:])
    Wres[0, ESN_param['resSize']-1] = ESN_param["cteRes"][0] * np.random.rand()

    # Spectral radius control
    if ESN_param['rhoControl'] == 1:
        rhoW = max(abs(np.linalg.eig(Wres)[0]))
        Wres *= ESN_param['spectralRadius'] / rhoW

    W = {'Win': Win, 'Wres': Wres, 'Wout': Wout, 'Wfed': Wfed}
    return W
##################
def pickupTwoNodes(weightNode,gamma):
# Auxiliar function selecting two nodes for adding an edge.
# each node has a related weight.
# It is not exactly power law but is made in order of being faster using np.searchsorted()
# this function return the index nearest to the value.
    a = np.searchsorted(weightNode, np.random.rand() * np.power(len(weightNode), 1/(gamma-1)))
    b = np.searchsorted(weightNode, np.random.rand() * np.power(len(weightNode), 1/(gamma - 1)))
    if a == len(weightNode):
        a = 0 # it is made circular, a kind of mod function
    if b == len(weightNode):
        b = 0  # it is made circular, a kind of mod function

    return a, b

# Scale-free network
def NetworkInitializationScaleFree(ESN_param):
    """
    Reservoir is initialized with a recurrent graph, it has the parameters for controlling spectral radius and density
    NOTE: Doesn't control exist of recurrences
    Inputs: ESN_param is a dictionary  with all the parameters for the ESN model.
    Returns: Win (input-reservoir weights) and W (reservoir weights), Wfb (feedback connections).
    """
    Win = ESN_param["cteIn"][0]*np.random.rand(ESN_param['resSize'], 1 + ESN_param['inSize']) - ESN_param["cteIn"][1]
    Wfed = ESN_param["cteFed"][0]*np.random.rand(ESN_param['resSize'], ESN_param['outSize']) - ESN_param["cteFed"][1]
    Wres = np.zeros(shape=(ESN_param['resSize'], ESN_param['resSize']))

    if ESN_param["BiasReservoirOut"] == 1:
        Wout = np.zeros(shape=(ESN_param['resSize'] + 2, ESN_param['outSize']))
    else:
        Wout = np.zeros(shape=(ESN_param['resSize'] + 1, ESN_param['outSize']))

    # Assign the reservoir weights
    gamma = 2.2  # from Physical letters 87-27
    weightNode = [np.power(i, 1/(gamma-1)) for i in range(ESN_param['resSize'])]
    counter = 0
    while counter < (ESN_param['density'] * np.power(ESN_param['resSize'], 2)):
        a, b = pickupTwoNodes(weightNode, gamma)
#       print("counter", counter,"Wres ", Wres[a,b],  " a:", a, "b: ", b)

        if a < 500 and b < 500 and Wres[a, b] == 0 and a != b:
            Wres[a, b] = ESN_param["cteRes"][0]*np.random.rand() - ESN_param["cteRes"][1]
            counter = counter + 1

    # control of spectra
    if ESN_param['rhoControl'] == 1:
        rhoW = max(abs(np.linalg.eig(Wres)[0]))
        Wres *= ESN_param['spectralRadius'] / rhoW

    W = {'Win': Win, 'Wres': Wres, 'Wout': Wout, 'Wfed': Wfed}
    return W
####
    # Density control
    #dens = np.random.rand(ESN_param['resSize'], ESN_param['resSize'])
    #if ESN_param['density'] < 1:  # density is 0.2 then 80% of elements are zeros
    #    Wres[np.where(dens > ESN_param['density'])] = 0
    # Spectral radius control
    ##############

DIGITS = 10000
def pi_digits (x):
    """
    Generate x digits of pi
    """
    k, a, b, a1, b1 = 2, 4, 1, 12, 4
    while x > 0:
        p, q, k = k*k, 2*k+1, k + 1
        a, b, a1, b1 = a1, b1, p*a+q*a1, p*b+q*b1
        d, d1 =a/b, a1/b1
        while d == d1 and x>0:
            yield int(d)
            x -= 1
            a, a1 = 10*(a % b), 10*(a1 % b1)
            d, d1 = a/b, a1/b1

def input_reservoir_sign(DIGITS):
    # Returns a np array with the sign of Pi digits.
    digits = [str(n) for n in pi_digits (DIGITS)]
    signs = np.zeros(shape=(1, len(digits)))+1
    for i in range(len(digits)):
        if int(digits[i])<=4:
            signs[0, i] = -1

    return signs

auxSign = input_reservoir_sign(DIGITS)
np.save("PiSigns.npy",auxSign)
#######################
#(Done)
def NetworkInitializationMartens(ESN_param):
    """
    Reservoir is initialized with a recurrent graph, it has the parameters for controlling spectral radius and density
    Inputs: ESN_param is a dictionary  with all the parameters for the ESN model.
    Returns: Win (input-reservoir weights) and W (reservoir weights), Wfb (feedback connections).
    Initialization was taken from "Deep learning via hessian-free approximation of James Martens.
    He said: "we hard limit the number of non-zero incoming connection weight to each unit (we used 15 in pur experiments)
    He said: it helps for avoiding saturation.  Then there is only 15 non-zero weights per row.
    # If controlSCR is 1 then
    """
    Martens_Constant = 15
    Win = ESN_param["cteIn"][0]*np.random.rand(ESN_param['resSize'], 1 + ESN_param['inSize']) - ESN_param["cteIn"][1]
    Wfed = ESN_param["cteFed"][0]*np.random.rand(ESN_param['resSize'], ESN_param['outSize']) - ESN_param["cteFed"][1]
    # Like Claudio suggested: identity -> 1 element outside -> permutation.
    Wres = ESN_param["cteFed"][0] * np.random.rand(ESN_param['resSize']) * np.identity(ESN_param['resSize'])
    Wres[ESN_param['resSize']-1, 0] = np.random.rand()
    Wres = np.random.permutation(Wres) # makes permutation among rows.

    if ESN_param["BiasReservoirOut"] == 1:
        Wout = np.zeros(shape=(ESN_param['resSize'] + 2, ESN_param['outSize']))
    else:
        Wout = np.zeros(shape=(ESN_param['resSize'] + 1, ESN_param['outSize']))

    auxRes = ESN_param["cteRes"][0] * np.random.rand(ESN_param['resSize'], Martens_Constant) - ESN_param["cteRes"][1]
    for i in range(ESN_param['resSize']):
        Wres[i, 0: Martens_Constant] = auxRes[i, :]
        Wres[i, :] = np.random.permutation(Wres[i, :])

    np.fill_diagonal(Wres, 0) # fill diag with 0s for avoiding loops.

    # to guarantee a cyrcle I assign a simple circle plus the Martens
#    if controlSCR == 1:
#        sequence = np.arange(ESN_param['resSize'])
#        lowerDiag = ((sequence+1)[0:(ESN_param['resSize']-1)], sequence[0:(ESN_param['resSize']-1)])
#        Wres[lowerDiag] = np.random.rand(1)[0]
#        Wres[0,ESN_param['resSize']-1] = np.random.rand(1)[0]

    # Spectral radius control
    if ESN_param['rhoControl'] == 1:
        rhoW = max(abs(np.linalg.eig(Wres)[0]))
        Wres *= ESN_param['spectralRadius'] / rhoW

    W = {'Win': Win, 'Wres': Wres, 'Wout': Wout, 'Wfed': Wfed}
    return W
#######################################
# (DONE)
def NetworkInitializationDLR(ESN_param, auxSign,v, r):
    """
    Reservoir is initialized as delay line reservoir (DLR). Roda, Tino in IEEE Transactions NNs
    Inputs: ESN_param is a dictionary  with all the parameters for the ESN model.
    Returns: Win (input-reservoir weights) and W (reservoir weights), Wfb (feedback connections).
    """
    # This structure has two new hyperparameters: the input-reservoir weight v and weight in the reservoir matrix.
    # All the elements in the lower diagonal has same value w.
    # The only randomness is in the sign of the input-reservoir weights, that authors applies psuedo-randomness sign.
    # Given Pi with d_1,d_2....d_N decimals, sign is minus if d_i in [0,4], d_i in [5,9]
    # The parameters v and w are given by function parameters.
    # It has added a dummy neuron by me, it is not mention about it on the paper of Tino and Rodan.

    Win = ESN_param["cteIn"][0] * np.random.rand(ESN_param['resSize'], 1 + ESN_param['inSize']) - ESN_param["cteIn"][1]
    Win[:, 0] = v*auxSign[0:ESN_param['resSize']] # all have same weight with different sign, according to the PI digit value.
    Wfed = ESN_param["cteFed"][0]*np.random.rand(ESN_param['resSize'], ESN_param['outSize']) - ESN_param["cteFed"][1]
    if ESN_param["BiasReservoirOut"] == 1:
        Wout = np.zeros(shape=(ESN_param['resSize'] + 2, ESN_param['outSize']))
    else:
        Wout = np.zeros(shape=(ESN_param['resSize'] + 1, ESN_param['outSize']))

    Wres = np.zeros(shape=(ESN_param['resSize'], ESN_param['resSize']))
    sequence = np.arange(ESN_param['resSize'])
    lowerDiag = ((sequence+1)[0:(ESN_param['resSize']-1)], sequence[0:(ESN_param['resSize']-1)])
    # when r is 0 means that I assign random values.
    if r == 0:
        Wres[lowerDiag] = np.random.rand(1)[0]
    else:
        Wres[lowerDiag] = r

    # Spectral radius control
    if ESN_param['rhoControl'] == 1:
        rhoW = max(abs(np.linalg.eig(Wres)[0]))
        Wres *= ESN_param['spectralRadius'] / rhoW

    W = {'Win': Win, 'Wres': Wres, 'Wout': Wout, 'Wfed': Wfed}
    return W
########################################
# (DONE)
def NetworkInitializationDLRB(ESN_param, auxSign,v, r, b):
    """
    Reservoir is initialized as delay line reservoir with backward connections (DLRB). Roda, Tino in IEEE Transactionson NNs
    Inputs: ESN_param is a dictionary  with all the parameters for the ESN model.
    Returns: Win (input-reservoir weights) and W (reservoir weights), Wfb (feedback connections).
    """
    Win = ESN_param["cteIn"][0] * np.random.rand(ESN_param['resSize'], 1 + ESN_param['inSize']) - ESN_param["cteIn"][1]
    Win[:,0] = v*auxSign[0:ESN_param['resSize']] # all have same weight with different sign, according to the PI digit value.
    Wfed = ESN_param["cteFed"][0]*np.random.rand(ESN_param['resSize'], ESN_param['outSize']) - ESN_param["cteFed"][1]
    if ESN_param["BiasReservoirOut"] == 1:
        Wout = np.zeros(shape=(ESN_param['resSize'] + 2, ESN_param['outSize']))
    else:
        Wout = np.zeros(shape=(ESN_param['resSize'] + 1, ESN_param['outSize']))

    Wres = np.zeros(shape=(ESN_param['resSize'], ESN_param['resSize']))
    sequence = np.arange(ESN_param['resSize'])
    lowerDiag = ((sequence+1)[0:(ESN_param['resSize']-1)], sequence[0:(ESN_param['resSize']-1)])
    upperDiag = (sequence[0:(ESN_param['resSize'] - 1)], (sequence + 1)[0:(ESN_param['resSize'] - 1)])

    # when r is 0 means that I assign random values.
    if r == 0:
        Wres[lowerDiag] = np.random.rand(1)[0]
        Wres[upperDiag] = np.random.rand(1)[0]
    else:
        Wres[lowerDiag] = r
        Wres[upperDiag] = b

    # Spectral radius control
    if ESN_param['rhoControl'] == 1:
        rhoW = max(abs(np.linalg.eig(Wres)[0]))
        Wres *= ESN_param['spectralRadius'] / rhoW

    W = {'Win': Win, 'Wres': Wres, 'Wout': Wout, 'Wfed': Wfed}
    return W
########################################
# (DONE)
def NetworkInitializationSCR(ESN_param,auxSign,v, r):
    """
    Reservoir is initialized as a simple circle reservoir (SRC). Roda, Tino in IEEE Transactionson NNs
    Inputs: ESN_param is a dictionary  with all the parameters for the ESN model.
    Returns: Win (input-reservoir weights) and W (reservoir weights), Wfb (feedback connections).

    Requires: auxSign=np.load("PiSigns.npy")
    v and r are constants the same for all.
    """
    Win = ESN_param["cteIn"][0] * np.random.rand(ESN_param['resSize'], 1 + ESN_param['inSize']) - ESN_param["cteIn"][1]
    Win[:,0] = v*auxSign[0:ESN_param['resSize']] # all have same weight with different sign, according to the PI digit value.
    Wfed = ESN_param["cteFed"][0]*np.random.rand(ESN_param['resSize'], ESN_param['outSize']) - ESN_param["cteFed"][1]
    if ESN_param["BiasReservoirOut"] == 1:
        Wout = np.zeros(shape=(ESN_param['resSize'] + 2, ESN_param['outSize']))
    else:
        Wout = np.zeros(shape=(ESN_param['resSize'] + 1, ESN_param['outSize']))

    Wres = np.zeros(shape=(ESN_param['resSize'], ESN_param['resSize']))
    sequence = np.arange(ESN_param['resSize'])
    lowerDiag = ((sequence+1)[0:(ESN_param['resSize']-1)], sequence[0:(ESN_param['resSize']-1)])

    # when r is 0 means that I assign random values.
    if r == 0:
        Wres[lowerDiag] = np.random.rand(1)[0]
        Wres[0,ESN_param['resSize']-1] = np.random.rand(1)[0]
    else:
        Wres[lowerDiag] = r
        Wres[0,ESN_param['resSize']-1] = r

    # Spectral radius control
    if ESN_param['rhoControl'] == 1:
        rhoW = max(abs(np.linalg.eig(Wres)[0]))
        Wres *= ESN_param['spectralRadius'] / rhoW

    W = {'Win': Win, 'Wres': Wres, 'Wout': Wout, 'Wfed': Wfed}
    return W


# To draw the reservoir graph:
#import networkx as nx
#G = nx.from_numpy_matrix(Wres, create_using=nx.DiGraph)
#layout = nx.spring_layout(G)
#nx.draw(G, layout)
#nx.draw_networkx_edge_labels(G, pos=layout)
#plt.show()