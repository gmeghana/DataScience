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
from B_process_data import *
from B_detect_bot import *

#Adjusted code from BIRDNEST author's source code to fit project tasking and add response time feature

np.set_printoptions(threshold='nan')
np.set_printoptions(linewidth=160)

#user mode or product mode
USE_PRODUCTS = 0
keyword = 'prod' if USE_PRODUCTS else 'user'

#Initializing first iteration
USE_TIMES = 1

#initializing number of clusters
K =6

#MAIN Function
# @profile
#import_data()
dataname = 'FacebookBids'

#PROCESS data
#temporal, usermap = load_bidder_data()
#(iat_arr, complete_iat_arr, rsp_arr, complete_rsp_arr, ids) = processing_data(temporal, dataname)
#at_arr = np.array(iat_arr)
#complete_iat_arr = np.array(complete_iat_arr)
#rsp_arr = np.array(rsp_arr)
#complete_rsp_arr = np.array(complete_rsp_arr)
#SAVE variables as pickle to save time
#pickle.dump((iat_arr, complete_iat_arr, rsp_arr, complete_rsp_arr, ids, usermap), open('%s_2bucketed.pickle' % (dataname), 'wb'))

#LOAD processed data
(iat_arr, complete_iat_arr, rsp_arr, complete_rsp_arr, ids, usermap) = pickle.load(open('%s_2bucketed.pickle' % (dataname), 'rb'))

#Cacluate up the normalized response time distribution of bots (bad) and all humans (good)
bad_time_distn = np.array([0]*complete_rsp_arr.shape[1], dtype=float)
bad_robot_iat_arr = []
bad_index = np.zeros((103,))
bad_dist_sum_axis_1 = np.zeros((103,))
good_time_distn = np.array([0]*complete_rsp_arr.shape[1], dtype=float)
good_hum_iat_arr = []
good_index = np.zeros((complete_rsp_arr.shape[0]-103,))
good_dist_sum_axis_1 = np.zeros((complete_rsp_arr.shape[0]-103,))
counter = 0
good_counter = 0
for id in ids:
    if int(float(usermap[id-1][1])) == 1:
        bad_index[counter] = id-1
        bad_dist_sum_axis_1[counter] = np.sum(complete_rsp_arr[id-1])
        bad_robot_iat_arr.append(complete_rsp_arr[id-1])
        if(bad_dist_sum_axis_1[counter]> 0): #since using complete_iat_arr then need to handle the situation when dividing by zero
            cur = (complete_rsp_arr[id-1] / np.sum(complete_rsp_arr[id-1]))
        else:
            cur = complete_rsp_arr[id-1]
        bad_time_distn += cur
        counter+=1
    else:
        good_index[good_counter] = id-1
        good_dist_sum_axis_1[good_counter] = np.sum(complete_rsp_arr[id-1])
        good_hum_iat_arr.append(complete_rsp_arr[id-1])
        if(good_dist_sum_axis_1[good_counter]> 0): #since using complete_iat_arr then need to handle the situation when dividing by zero
            cur = (complete_rsp_arr[id-1] / np.sum(complete_rsp_arr[id-1]))
        else:
            cur = complete_rsp_arr[id-1]
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
#iat_sums = iat_arr.sum(axis=1)

bad_time_norm = bad_robot_iat_arr / bad_iat_sums[:, np.newaxis]
good_time_norm = good_hum_iat_arr / good_iat_sums[:, np.newaxis]

#time_norm = iat_arr / iat_sums[:, np.newaxis]
bad_rsp_hist = bad_time_norm.sum(axis=0)
bad_rsp_hist = bad_rsp_hist / np.sum(bad_rsp_hist)

good_rsp_hist = good_time_norm.sum(axis=0)
good_rsp_hist = good_rsp_hist / np.sum(good_rsp_hist)

#FIND anomalous behavior and output scores of each user
suspect = detect_bot(complete_rsp_arr, complete_iat_arr, USE_TIMES, K)


###############################################################
NUM_TO_OUTPUT = 1500
#Sort scores from highest to lowest and output the outcome in the following files
susp_sorted = np.array([(x[0]) for x in sorted(enumerate(suspect), key=itemgetter(1), reverse=True)])
most_susp = susp_sorted[range(len(susp_sorted))]
with open('%s_top%d_ids.txt' % (dataname, NUM_TO_OUTPUT), 'w') as outfile:
    with open('%s_top%d_scores.txt' % (dataname, NUM_TO_OUTPUT), 'w') as out_scores:
        with open('%s_top%d_rsp.txt' % (dataname, NUM_TO_OUTPUT), 'w') as out_rsp:
            with open('%s_top%d_iat2.txt' % (dataname, NUM_TO_OUTPUT), 'w') as out_iat:
                for i in most_susp:
                    if usermap == None:
                        print >> outfile, '%s' % (ids[i], )
                    else:
                        print >> outfile, '%s %s %s' % (ids[i], usermap[ids[i]-1][0], usermap[ids[i]-1][1])
                    print >> out_scores, '%d %f' % (ids[i], suspect[i])
                    print >> out_rsp, complete_rsp_arr[i,:]
                    print >> out_iat, complete_iat_arr[i,:]

#Calculate the distributions of those that were considered "bad" (in position 1 - 103 since there are 103 bots in data set) and those that were considered "god" (103-end)
TOP_N_SUSPECTS = 103
bad = susp_sorted[range(TOP_N_SUSPECTS)]
bad_time_ave = np.array([0]*complete_iat_arr.shape[1], dtype=float)
good_time_ave = np.array([0]*complete_iat_arr.shape[1], dtype=float)
bad_dt_ave = np.array([0]*complete_rsp_arr.shape[1], dtype=float)
good_dt_ave = np.array([0]*complete_rsp_arr.shape[1], dtype=float)
for i in range(len(suspect)):
    #cur = (iat_arr[i,:] / np.sum(iat_arr[i,:]))
    if(np.sum(complete_iat_arr[i,:]) > 0): #since using complete_iat_arr then need to handle the situation when dividing by zero
        cur = (complete_iat_arr[i,:] / np.sum(complete_iat_arr[i,:]))
    else:
        cur = complete_iat_arr[i,:]

    if(np.sum(complete_rsp_arr[i,:]) > 0): #since using complete_iat_arr then need to handle the situation when dividing by zero
        cur2 = (complete_rsp_arr[i,:] / np.sum(complete_rsp_arr[i,:]))
    else:
        cur2 = complete_rsp_arr[i,:]
    if i in bad:
        bad_time_ave += cur
        bad_dt_ave += cur2
    else:
        good_time_ave += cur
        good_dt_ave += cur2

#Calculate the normalized response time distibution and IAT distribution of all users
iat_sums = complete_iat_arr.sum(axis=1)
rsp_sums = complete_rsp_arr.sum(axis=1)

for i in range(len(iat_sums)):
    if iat_sums[i] == 0:
        iat_sums[i] = 1 #keep from dividing by zero when normalizing

for i in range(len(rsp_sums)):
    if rsp_sums[i] == 0:
        rsp_sums[i] = 1
time_norm = complete_iat_arr / iat_sums[:, np.newaxis]
dt_norm = complete_rsp_arr / rsp_sums[:, np.newaxis]

iat_hist = time_norm.sum(axis=0)
dt_hist = dt_norm.sum(axis=0)
iat_hist = iat_hist / np.sum(iat_hist)
dt_hist = dt_hist / np.sum(dt_hist)

#Plot the response time distirbutions of those that were considered "good" and the behavior of users considered "bad"
tx = range(1, len(range(len(complete_iat_arr[1,:])+1)))
dtx = range(1, len(range(len(complete_rsp_arr[1,:])+1)))

fig = plt.figure(figsize=(8, 5))
plt.subplot(1,2,1)
plt.bar(dtx, good_dt_ave, color='purple')
plt.title('Normal Response Time of Users', size=18)

plt.subplot(1,2,2)
plt.bar(dtx, bad_dt_ave, color='red')
plt.title('Detected users', size=18)
plt.xlabel('Response Time (bucketized)', size=14)
fig.text(0.5, 0.02, 'Response Time bucket', ha='center', size=18)

plt.savefig('Facebook_rsp_tm_goodbad.png')

#Plot the IAT distirbutions of those that were considered "good" and the behavior of users considered "bad"
fig = plt.figure(figsize=(8, 5))
plt.subplot(1,2,1)
plt.bar(tx, good_time_ave, color='green')
plt.title('Normal users', size=18)

plt.subplot(1,2,2)
plt.bar(tx, bad_time_ave, color='red')
plt.title('Detected users', size=18)
plt.xlabel('Time between bids (bucketized)', size=14)
fig.text(0.5, 0.02, 'IAT bucket', ha='center', size=18)

plt.savefig('Facebook_iat2_goodbad.png')

buckets = np.arange(complete_rsp_arr.shape[1])
#Plot all users response time distribution - normalized
plt.hold(False)
fig = plt.figure()
plt.bar(buckets, dt_hist, color='blue')
plt.title('All users normalized')
plt.xlabel('Response Time buckets')
plt.ylabel('Frequency')
plt.savefig('Distribution_all_users_RSP_IAT_.png')

#Plot distribution of  bot response time distributions vs true human response time distributions
plt.hold(False)
fig = plt.figure(figsize=(8, 5))
plt.subplot(1,2,1)
plt.bar(buckets, good_rsp_hist, color='blue')
plt.title('All Humans Normalized')
plt.xlabel('Response Time buckets')
plt.ylabel('Frequency')
plt.ylim(ymax=.15,ymin=0)

plt.subplot(1,2,2)
plt.bar(buckets, bad_rsp_hist, color='red')
plt.title('All Bots Normalized')
plt.xlabel('Response Times buckets')
plt.ylim(ymax=.15,ymin=0)

plt.savefig('Distribution_norm_bot_users_RSP_IAT.png')