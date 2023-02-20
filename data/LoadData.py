#################
# Load the data
#################
import numpy as np
#data_path = "/Users/Seba/PycharmProjects/Data/"
#data_path = "C:\\Users\\Kofola\\PycharmProjects\\EvoESN_V1\\Data\\"
#data_path = "C:/Users/Kofola/PycharmProjects/EvoESN_V1/Data/"
#################
# Lorenz system
# already scaled
dataLorenz = np.load('C:/Users/Kofola/PycharmProjects/EvoESN_V1/Data/LorenzScaledXdata.npy')
# Mackey-Glass system
dataMGS = np.load('C:/Users/Kofola/PycharmProjects/EvoESN_V1/Data/mackey_glass_t17.npy')
#
#np.random.seed(1)  # seed random number generator
#trainLen = 3000
#testLen = 100
#initLen = 2000 # from what point starts the training
#inSize = outSize = 1
#data = np.loadtxt(data_path+'MackeyGlass_t17.txt')
###########
# Henon train
#dataHennon = np.loadtxt('C:/Users/Kofola/PycharmProjects/EvoESN_V1/Data/HenonTrain.txt')
#data = data[1:,0];
#np.random.seed(1)  # seed random number generator
#trainLen = 2500
#testLen = 1400
#initLen = 1
#inSize = outSize = 1
###############
# Lorenz train
#data = np.loadtxt(data_path+'Lorenz.dat')
#np.random.seed(1)  # seed random number generator
#trainLen = 5000
#testLen = 4000
#initLen = 1
#inSize = outSize = 1
#################
# Daily total sunspot number
# http://www.sidc.be/silso/datafiles
#monthly
#my_dataMonth = genfromtxt(r"C:\Users\Kofola\PycharmProjects\EvoESN_V1\data\SN_m_tot_V2.0.csv",delimiter=";")
# daily
#my_dataDay = genfromtxt(r"C:\Users\Kofola\PycharmProjects\EvoESN_V1\data\SN_d_tot_V2.0.csv",delimiter=";")
#data = np.loadtxt(data_path+'SN_d_tot_V2.0.txt')
# it is important only column 4.
#data=data[:,4]
#maxD=np.max(data)
#minD=np.min(data)
#data=(data-minD)*(1/(maxD-minD))
#np.random.seed(1)  # seed random number generator
#trainLen = 10000
#testLen = 1000
#initLen = 1
#inSize = outSize = 1
