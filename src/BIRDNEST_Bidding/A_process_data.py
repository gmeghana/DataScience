import numpy as np
import csv
import math
import operator
import cPickle as pickle
import sys
import pandas as pd

#Adjusted from BIRDNEST author's source code to fit project purposes

TIME_LOG_BASE = 2 #Changed from 5

def load_bidder_data():
    #Take data from csv file to sequence/lists to later process into histograms
    csvfile = open('../Data/train_seq_bidder_id_time.csv', 'rb')
    outfile = open('preproc_out.txt', 'w')
    train_tbl = csv.reader(csvfile, delimiter=',', escapechar='\\', quotechar=None)

    iat = {} # {user: [auction, time]}
    usermap = list() #{user: (bidder_id, label)}
    range_user = range(1,4000000,1) #user IDs will go from 1 to max with no gap
    counter = 0
    how_many_total_bots = 0
    for toks in train_tbl:
        username, label, auction, time = [toks[0], toks[3], toks[5], int(toks[8])]
        if (username, label) not in usermap:
            bidder = range_user[counter]
            usermap.append((username, label))
            counter += 1
            if int(float(label)) == 1:
                how_many_total_bots +=1
        if bidder not in iat:
            iat[bidder] = []
        iat[bidder].append((auction, time))
    print >> outfile, '%s bidders before taking out users with not enough data and %s total bots in training set' %(bidder, how_many_total_bots)
    return iat, usermap #, to_corr


def processing_data(iat, dataname):
    complete_iat_arr = []
    iat_arr = []
    ids = []
    max_time_diff = -1
    #find the max time difference for the log-based IAT binning
    for user in iat:
        cur_iat = sorted(iat[user], key=operator.itemgetter(1))
        for i in range(1, len(cur_iat)):
            time_diff = cur_iat[i][1] - cur_iat[i-1][1]
            max_time_diff = max(max_time_diff, time_diff)

    #S = number of buckets
    S = int(1 + math.floor(math.log(1 + max_time_diff, TIME_LOG_BASE)))
    for user in iat:
        iat_counts = [0] * S
        if len(iat[user]) <= 1: #if only one bid then just append an all zero distribution (iat_arr is with pruning and complete_iat_arr is with these zero distributions)
            complete_iat_arr.append(iat_counts)
            ids.append(user) #take out if you want only users that have enough data
            continue
        cur_iat = sorted(iat[user], key=operator.itemgetter(1))
        for i in range(1, len(cur_iat)):
            time_diff = cur_iat[i][1] - cur_iat[i-1][1] #Calculate IAT
            iat_bucket = int(math.floor(math.log(1 + time_diff, TIME_LOG_BASE))) #Find what bucket it is suppose to go in
            iat_counts[iat_bucket] += 1 #increase the bucket
        iat_arr.append(iat_counts) #append the histogram to iat_arr
        ids.append(user)
        complete_iat_arr.append(iat_counts)


    with open('%s_iat_bucketed.txt' % (dataname), 'w') as iat_file:
        for row in complete_iat_arr: #changed from iat_arr since want to keep users that do not have IATs but may have other features
            print >> iat_file, ' '.join([str(x) for x in row])

    iat_arr = np.array(iat_arr)
    return (iat_arr, complete_iat_arr, ids)

def main(): #run this to debug A_process_data functions
    iat, usermap = load_bidder_data()
    processing_data(iat, 'Facebook_bids')

if __name__=='__main__':
    main()
