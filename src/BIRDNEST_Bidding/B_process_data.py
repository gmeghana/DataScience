import numpy as np
import csv
import math
import operator
import cPickle as pickle
import sys
import pandas as pd

#Based on BIRDNEST author's source code but adjusted to fit project and newest feature of response time

TIME_LOG_BASE = 2 #Changed from 5

def load_bidder_data():
    #Take data from csv file to sequence/lists to later process into histograms
    csvfile = open('../Data/train_iat_rsp6.csv', 'rb')
    outfile = open('preproc_out.txt', 'w')
    train_tbl = csv.reader(csvfile, delimiter=',', escapechar='\\', quotechar=None)

    temporal = {} # {user: [auction, time, dt]}
    usermap = list() #{user: (bidder_id, label)}
    range_user = range(1,4000000,1) #user IDs will go from 1 to max with no gap
    counter = 0
    how_many_total_bots = 0
    for toks in train_tbl:
        if toks[4] == 'dt': #toks[12]
            continue
        if toks[4] == 'NULLVALUE': #toks[12]
            username, label, auction, time, dt = [toks[0], toks[3], toks[1], int(toks[2]), toks[4]]
        else:
            username, label, auction, time, dt = [toks[0], toks[3], toks[1], int(toks[2]), float(toks[4])] #[toks[0], toks[3], toks[5], int(toks[8]), toks[12]]
        if (username, label) not in usermap:
            bidder = range_user[counter]
            usermap.append((username, label))
            counter += 1
            if int(float(label)) == 1:
                how_many_total_bots +=1
        if bidder not in temporal:
            temporal[bidder] = []
        temporal[bidder].append((auction, time, dt))
    print >> outfile, '%s bidders before taking out users with not enough data and %s total bots in training set' %(bidder, how_many_total_bots)
    return temporal, usermap


def processing_data(temporal, dataname):
    complete_iat_arr = []
    complete_rsp_arr = []
    iat_arr = []
    rsp_arr = []
    ids = []
    max_time_diff = -1
    max_dt_diff = -1
    num_null_values = 0
    #find the max time difference for the log-based IAT binning and log-based RSP binning
    for user in temporal:
        cur_iat = sorted(temporal[user], key=operator.itemgetter(1))
        #IAT
        for i in range(1, len(cur_iat)):
            time_diff = cur_iat[i][1] - cur_iat[i-1][1]
            max_time_diff = max(max_time_diff, time_diff)
        #Response Time (dt)
        for i in range(0, len(cur_iat)):
            if cur_iat[i][2] == 'NULLVALUE':
                num_null_values +=1 #just ignore the case where the dt is NULL (this is because it is the initiator of the auction thus no response time
                #dt = 0.0 #this would assume that the response time of an initiator of an auction is 0.0 instead of ignored.
            else:
                dt = cur_iat[i][2]
                max_dt_diff = max(max_dt_diff, dt)

    #S = number of time buckets
    #R = number of dt buckets
    S = int(1 + math.floor(math.log(1 + max_time_diff, TIME_LOG_BASE)))
    R = int(1 + math.floor(math.log(1 + max_dt_diff, TIME_LOG_BASE)))
    for user in temporal:
        iat_counts = [0] * S
        rsp_counts = [0] * R
        cur_iat = sorted(temporal[user], key=operator.itemgetter(1))
        for i in range(1, len(cur_iat)):
            time_diff = cur_iat[i][1] - cur_iat[i-1][1] #Calculate IATs for user
            iat_bucket = int(math.floor(math.log(1 + time_diff, TIME_LOG_BASE))) #Find the bucket to place the IAT calculation
            iat_counts[iat_bucket] += 1 #increase the count of that bucket
        for j in range(0, len(cur_iat)): #bucketize dt
            if cur_iat[j][2] == 'NULLVALUE': #if there is no response time then we are not going to bucket anything
                num_null_values -=1 #this should go down to zero and be the same
                #dt = 0.0 #use this if the assumption is that no response time is really a 0.0 response time.
            else:
                dt = cur_iat[j][2] #response time comes in with the table (reflects Yellow Duck response time feature)
                rsp_bucket = int(math.floor(math.log(1 + dt, TIME_LOG_BASE))) #find which bucket the response time fits in
                rsp_counts[rsp_bucket] += 1 #Then increase the bucket count
        iat_arr.append(iat_counts)
        rsp_arr.append(rsp_counts)
        ids.append(user)
        complete_iat_arr.append(iat_counts)
        complete_rsp_arr.append(rsp_counts)

    #save histograms in a text file
    with open('%s_rsp_bucketed.txt' % (dataname), 'w') as rsp_file:
        for row in rsp_arr:
                print >> rsp_file, ' '.join([str(x) for x in row])

    with open('%s_iat2_bucketed.txt' % (dataname), 'w') as iat_file:
        for row in complete_iat_arr: #changed from iat_arr since want to keep users that do not have IATs but may have other features
            print >> iat_file, ' '.join([str(x) for x in row])

    iat_arr = np.array(iat_arr)
    rsp_arr = np.array(rsp_arr)
    complete_iat_arr = np.array(complete_iat_arr)
    complete_rsp_arr = np.array(complete_rsp_arr)
    return (iat_arr, complete_iat_arr, rsp_arr, complete_rsp_arr, ids)

#function for debugging process_data functions
def main():
    iat, usermap = load_bidder_data()
    processing_data(iat, 'Facebook_bids')

if __name__=='__main__':
    main()
