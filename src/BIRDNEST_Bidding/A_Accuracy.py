from __future__ import division
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import cross_validation
from collections import Counter
import math
from operator import itemgetter
import operator
import random
import numpy as np
import sklearn
from sklearn.cluster import KMeans
from scipy.special import gammaln, psi
import cPickle as pickle
import matplotlib.pyplot as plt
from A_process_data import *
from A_detect_bot import *


#Purpose: Calculate ROC and AUC to compare against other algorithms

np.set_printoptions(threshold='nan')
np.set_printoptions(linewidth=160)

if __name__=='__main__':
    #MAIN Function
    #import_data()
    dataname = 'FacebookBids'

    TOP_N_SUSPECTS = 103 #total number of bots in training sample = 103

    #Initializing the flag to normalize the scores
    USE_TIMES = 1

    #PROCESS data
    #iat, usermap = load_bidder_data()
    #(iat_arr, complete_iat_arr, ids) = processing_data(iat, dataname)
    #iat_arr = np.array(iat_arr)
    #complete_iat_arr = np.array(complete_iat_arr)
    #pickle to save time
    #pickle.dump((iat_arr, complete_iat_arr, ids, usermap), open('%s_bucketed.pickle' % (dataname), 'wb'))
    #LOAD processed data
    (iat_arr, complete_iat_arr, ids, usermap) = pickle.load(open('%s_bucketed.pickle' % (dataname), 'rb'))

    #how many clusters to average accuracy over
    Num_clusters = 21
    x_K = np.arange(1,Num_clusters)
    y_auc = np.zeros((len(x_K),))
    avg_user_score = np.zeros(len(complete_iat_arr))
    v_time = np.zeros((len(x_K),))

    #initializing number of clusters
    for K in range(1,Num_clusters):
        #suspects, non_suspects = runmain(iat_arr, K) #with only users that have enough data for IAT distribution makes the accuracy worse
        suspects = detect_bot(complete_iat_arr, USE_TIMES, K)

        NUM_TO_OUTPUT = 1500
        #sort the scores from highest to lowest to get most_susp
        susp_sorted = np.array([(x[0]) for x in sorted(enumerate(suspects), key=itemgetter(1), reverse=True)])
        most_susp = susp_sorted[range(len(susp_sorted))]

        #Step for averaging user's scores (division outside of this for loop)
        for i in most_susp:
            avg_user_score[ids[i]-1] += suspects[i]

        #calculate ROC and AUC for current run with K clusters
        ranking = np.zeros(len(complete_iat_arr))
        counter = 0
        for i in most_susp:
            ranking[counter] = usermap[i][1]
            counter+=1
        #calculate ROC curve tpr and fpr
        tpr = np.zeros(len(complete_iat_arr))
        fpr = np.zeros(len(complete_iat_arr))
        total_ones = sum(ranking[:]==1)
        total_zeros = len(complete_iat_arr)-total_ones
        #creating a ROC curve
        for i in range(len(complete_iat_arr)):
            num_of_ones = sum(ranking[0:i])
            num_of_zeros = (i+1)-num_of_ones
            tpr[i] = num_of_ones/total_ones
            fpr[i] = num_of_zeros/total_zeros

        y_auc[K-1] = sklearn.metrics.auc(fpr,tpr)

    avg_user_score = avg_user_score/(Num_clusters-1)

    #rerank the users according to average scores
    avg_susp_sorted = np.array([(x[0]) for x in sorted(enumerate(avg_user_score), key=itemgetter(1), reverse=True)])
    most_susp = avg_susp_sorted[range(len(avg_susp_sorted))]

    #calculate ROC
    ranking = np.zeros(len(complete_iat_arr))
    counter = 0
    for i in most_susp:
        ranking[counter] = usermap[i][1]
        counter+=1

    tpr = np.zeros(len(complete_iat_arr))
    fpr = np.zeros(len(complete_iat_arr))
    total_ones = sum(ranking[:]==1)
    total_zeros = len(complete_iat_arr)-total_ones
    #creating a ROC curve
    for i in range(len(complete_iat_arr)):
        num_of_ones = sum(ranking[0:i])
        num_of_zeros = (i+1)-num_of_ones
        tpr[i] = num_of_ones/total_ones
        fpr[i] = num_of_zeros/total_zeros
    #Calculate AUC
    avg_auc = sklearn.metrics.auc(fpr,tpr)
    print avg_auc

    #plot ROC
    plt.figure()
    plt.plot(fpr,tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Average BIRDNEST ROC for IAT Only - K=1-20 clusters")
    plt.savefig("ROC_iat_avg.png")



    print 'Finished'
