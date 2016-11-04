import time
import numpy as np
import random
import matplotlib.pyplot as plt
import cPickle as pickle
from B_process_data import *
from B_detect_bot import *

#Adjusted from BIRDNEST source code for purposes of project
#Used to figure out the average runtime - Test scalability with data set (Adjusted from BIRDNEST author's source code)

dataname = 'FacebookBids'
#temporal, usermap = load_bidder_data()
#(iat_arr, complete_iat_arr, rsp_arr, complete_rsp_arr, ids) = processing_data(temporal, dataname)
(iat_arr, complete_iat_arr, rsp_arr, complete_rsp_arr, ids, usermap) = pickle.load(open('%s_2bucketed.pickle' % (dataname), 'rb'))

complete_iat_arr = np.array(complete_iat_arr)
complete_rsp_arr = np.array(complete_rsp_arr)
#Determine what fractions of the data set we are going to test for time
sampleFracs = (0.7)**np.array(range(0,20))
numBids = [0] * len(sampleFracs)
timeTaken = [0] * len(sampleFracs)
#Will also be viewing accuracy by varying time
y_auc = [0] * len(sampleFracs)
m = complete_iat_arr.shape[0]
numIters = 1
USE_TIMES = 1
#Number of clusters 1-Num_clusters
Num_clusters = 21
x_K = np.arange(1,Num_clusters)
y_K = np.zeros((Num_clusters,))

for K in x_K:
    for i in range(len(sampleFracs)):
        print "sampleFrac: ", sampleFracs[i]
        for j in range(numIters):
            sampleIdx = random.sample(range(m), int(round(m * sampleFracs[i])))
            iat_sub = complete_iat_arr[sampleIdx,:]
            rsp_sub = complete_rsp_arr[sampleIdx,:]

            #calculate run time
            timeBefore = time.time()
            suspects = detect_bot(complete_rsp_arr, complete_iat_arr, USE_TIMES, K)
            timeTaken[i] += (time.time() - timeBefore) / numIters

            #Calculate how many bids were used to create mixture model
            numBids[i] += len(rsp_sub)

        numBids[i] = numBids[i]/numIters #taking an average of number of bids used to create mixture model
    #average the runtime of all runs for when there are K clusters
    y_K[K-1] = np.mean(np.array(timeTaken))

#Plot the scalability
plt.figure(figsize=(6,6))
plt.loglog(numBids[:-6], timeTaken[:-6], '-o')
plt.loglog([1e4,1e7], [1,1e3], '-', color='black', linewidth=2.0)
plt.xlabel('Number of bids')
plt.ylabel('Time taken (s)')
plt.title('Scalability of BIRD')
plt.rcParams.update({'font.size': 24})
plt.xlim([1e4,4e6])
plt.ylim([1, 4e2])
plt.savefig('Scalability_timing_iat_rsp.png', bbox_inches='tight')
plt.show()

#plot the runtime
plt.figure()
plt.plot(x_K, y_K)
plt.xlabel('Number of Clusters')
plt.ylabel('Time taken (s)')
plt.title('Runtime of BIRD')
plt.rcParams.update({'font.size': 24})
plt.xlim([1e4,4e6])
plt.ylim([1, 4e2])
plt.savefig('Runtime_iat_rsp.png', bbox_inches='tight')
plt.show()
