from __future__ import division
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
from process_data_resp_tm import *

#Adjusted from BIRDNEST author's source code to fit project purposes

np.set_printoptions(threshold='nan')
np.set_printoptions(linewidth=160)

#user mode or product mode
USE_PRODUCTS = 0
keyword = 'prod' if USE_PRODUCTS else 'user'

#Initializing first iteration
USE_TIMES = 1

#initializing number of clusters
K = 5

#MAIN Function
# @profile
#import_data()
dataname = 'FacebookBids'

#PROCESS data
#iat, usermap = load_bidder_data()
#(iat_arr, complete_iat_arr, ids) = processing_data(iat, dataname)
#iat_arr = np.array(iat_arr)
#complete_iat_arr = np.array(complete_iat_arr)
#SAVE variables as pickle to save time
#pickle.dump((iat_arr, complete_iat_arr, ids, usermap), open('%s_bucketed.pickle' % (dataname), 'wb'))

#LOAD processed data
(iat_arr, complete_iat_arr, ids, usermap) = pickle.load(open('%s_bucketed.pickle' % (dataname), 'rb'))

#Show bad vs good feature distributions
bad_time_distn = np.array([0]*complete_iat_arr.shape[1], dtype=float)
bad_robot_iat_arr = []
bad_index = np.zeros((103,))
bad_dist_sum_axis_1 = np.zeros((103,))
good_time_distn = np.array([0]*complete_iat_arr.shape[1], dtype=float)
good_hum_iat_arr = []
good_index = np.zeros((complete_iat_arr.shape[0]-103,))
good_dist_sum_axis_1 = np.zeros((complete_iat_arr.shape[0]-103,))
counter = 0
good_counter = 0
for id in ids:
    if int(float(usermap[id-1][1])) == 1:
        bad_index[counter] = id-1
        bad_dist_sum_axis_1[counter] = np.sum(complete_iat_arr[id-1])
        bad_robot_iat_arr.append(complete_iat_arr[id-1])
        if(bad_dist_sum_axis_1[counter]> 0): #since using complete_iat_arr then need to handle the situation when dividing by zero
            cur = (complete_iat_arr[id-1] / np.sum(complete_iat_arr[id-1]))
        else:
            cur = complete_iat_arr[id-1]
        bad_time_distn += cur
        counter+=1
    else:
        good_index[good_counter] = id-1
        good_dist_sum_axis_1[good_counter] = np.sum(complete_iat_arr[id-1])
        good_hum_iat_arr.append(complete_iat_arr[id-1])
        if(good_dist_sum_axis_1[good_counter]> 0): #since using complete_iat_arr then need to handle the situation when dividing by zero
            cur = (complete_iat_arr[id-1] / np.sum(complete_iat_arr[id-1]))
        else:
            cur = complete_iat_arr[id-1]
        good_time_distn += cur
        good_counter+=1

bad_iat_sums = bad_dist_sum_axis_1
for i in range(len(bad_iat_sums)):
    if bad_iat_sums[i] == 0:
        bad_iat_sums[i] = 1 #keep from dividing by zero when normalizing

good_iat_sums = good_dist_sum_axis_1
for i in range(len(good_iat_sums)):
    if good_iat_sums[i] == 0:
        good_iat_sums[i] = 1

bad_time_norm = bad_robot_iat_arr / bad_iat_sums[:, np.newaxis]
good_time_norm = good_hum_iat_arr / good_iat_sums[:, np.newaxis]

bad_iat_hist = bad_time_norm.sum(axis=0)
bad_iat_hist = bad_iat_hist / np.sum(bad_iat_hist)

good_iat_hist = good_time_norm.sum(axis=0)
good_iat_hist = good_iat_hist / np.sum(good_iat_hist)


#FIND anomalous behavior and output scores of each user
suspect = detect_bot(complete_iat_arr, USE_TIMES, K)



###############################################################
NUM_TO_OUTPUT = 1500
#sort the users' scores from highest to lowest
susp_sorted = np.array([(x[0]) for x in sorted(enumerate(suspect), key=itemgetter(1), reverse=True)])
most_susp = susp_sorted[range(len(susp_sorted))]
#create text files to show ranking and distributions in the ranking from anomalous to least anomalous
with open('%s_top%d_ids.txt' % (dataname, NUM_TO_OUTPUT), 'w') as outfile:
    with open('%s_top%d_scores.txt' % (dataname, NUM_TO_OUTPUT), 'w') as out_scores:
        with open('%s_top%d_iat.txt' % (dataname, NUM_TO_OUTPUT), 'w') as out_iat:
            for i in most_susp:
                if usermap == None:
                    print >> outfile, '%s' % (ids[i], )
                else:
                    print >> outfile, '%s %s %s' % (ids[i], usermap[ids[i]-1][0], usermap[ids[i]-1][1])
                print >> out_scores, '%d %f' % (ids[i], suspect[i])
                print >> out_iat, complete_iat_arr[i,:]
#calculate the distributions of the supposed "good" and "bad" distributions based on bad = 1-103 most anomalous users and good = 103-end
TOP_N_SUSPECTS = 103
bad = susp_sorted[range(TOP_N_SUSPECTS)]
bad_time_ave = np.array([0]*complete_iat_arr.shape[1], dtype=float)
good_time_ave = np.array([0]*complete_iat_arr.shape[1], dtype=float)
for i in range(len(suspect)):
    if(np.sum(complete_iat_arr[i,:]) > 0): #since using complete_iat_arr then need to handle the situation when dividing by zero
        cur = (complete_iat_arr[i,:] / np.sum(complete_iat_arr[i,:]))
    else:
        cur = complete_iat_arr[i,:]
    if i in bad:
        bad_time_ave += cur
    else:
        good_time_ave += cur
#Calculates the normalized distribution of all users
iat_sums = complete_iat_arr.sum(axis=1)
for i in range(len(iat_sums)):
    if iat_sums[i] == 0:
        iat_sums[i] = 1 #keep from dividing by zero when normalizing
time_norm = complete_iat_arr / iat_sums[:, np.newaxis]
iat_hist = time_norm.sum(axis=0)
iat_hist = iat_hist / np.sum(iat_hist)

#Plot good and bad distributions based on algorithm output
tx = range(1, len(range(len(complete_iat_arr[1,:])+1)))
fig = plt.figure(figsize=(8, 5))
plt.subplot(1,2,1)
plt.bar(tx, good_time_ave, color='green')
plt.title('Normal users', size=18)

plt.subplot(1,2,2)
plt.bar(tx, bad_time_ave, color='red')
plt.title('Detected users', size=18)
plt.xlabel('Time between bids (bucketized)', size=14)
fig.text(0.5, 0.02, 'IAT bucket', ha='center', size=18)

plt.savefig('Facebook_iat_goodbad.png')

buckets = np.arange(complete_iat_arr.shape[1])
#plot all users noramlzied distribution
plt.hold(False)
fig = plt.figure()
plt.bar(buckets, iat_hist, color='blue')
plt.title('All users normalized')
plt.xlabel('IAT buckets')
plt.ylabel('Frequency')
plt.savefig('Distribution_all_users_IAT_only')

#plot IAT distribution of the true bots and the true normal people
plt.hold(False)
fig = plt.figure(figsize=(8, 5))
plt.subplot(1,2,1)
plt.bar(buckets, good_iat_hist, color='blue')
plt.title('All Humans Normalized')
plt.xlabel('IAT buckets')
plt.ylabel('Frequency')
plt.ylim(ymax=.15,ymin=0)

plt.subplot(1,2,2)
plt.bar(buckets, bad_iat_hist, color='red')
plt.title('All Bots Normalized')
plt.xlabel('IAT buckets')
plt.ylim(ymax=.15,ymin=0)

plt.savefig('Distribution_norm_bot_users_IAT_only')