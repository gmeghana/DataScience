__author__ = "andrew"

import numpy as np
import csv
import math
import operator
import cPickle as pickle
import os
from sklearn.preprocessing import normalize

"""
INDICES IN THE ORIGINAL DATAFILE
"""
USERID = 0
BOTLABEL = 3
AUCTIONID = 5
CATEGORY = 6
DEVICE = 7
TIMESTAMP = 8
IP = 9

"""
bidTuples format
(int) userNum,
(str) auctionId,
(int) time,
(str) category,
(str) device,
(str) ip
"""
BT_USERNUM = 0
BT_AUCTIONID = 1
BT_TIME = 2
BT_CATEGORY = 3
BT_DEVICE = 4
BT_IP = 5

TIME_LOG_BASE = 2
DATAHOME = "./data/"

def generate_mean_iats(bidTuples, userList, force=False):
    iats, maxVal = generate_iats(bidTuples, userList, force=force)
    return mean_by_user(iats)


def generate_mean_rts(bidTuples, userList, force=False):
    rts, auctionList, maxVal = generate_rts(bidTuples, userList, force=force)
    return mean_by_user(rts)

def generate_binned_rts(bidTuples, userList, force=False):
    if (not force) and os.path.isfile(DATAHOME+'rts_buckets.p'):
        print "rts_buckets file located. Loading..."
        rts_buckets = pickle.load(open(DATAHOME+'rts_buckets.p',"rb"))
    else:
        rts, auctionList, maxResponseTime = generate_rts(bidTuples, userList)
        rts_buckets = bucketize(rts, TIME_LOG_BASE, outfile=DATAHOME+"rt_buckets.txt", maxVal=maxResponseTime)
    return rts_buckets

def generate_binned_iats(bidTuples, userList, force=False):
    if (not force) and os.path.isfile(DATAHOME+'iats_buckets.p'):
        print "iats_buckets file located. Loading..."
    else:
        iats, maxIat = generate_iats(bidTuples, userList)
        iats_buckets = bucketize(iats, TIME_LOG_BASE, outfile=DATAHOME+"iat_buckets.txt", maxVal=maxIat)
    return iats_buckets


def generate_iats(bidTuples, userList, force=False):
    if (not force) and os.path.isfile(DATAHOME+'userData.p'):
        print "userData file located. Loading..."
        userData = pickle.load(open(DATAHOME+'userData.p',"rb"))
    else:
        # Generate userData, a list of times for each user
        userData = [[] for user in userList]
        for toks in bidTuples:
            userNum, auction, time = [toks[0], toks[1], toks[2]]
            userData[userNum].append(time)
    iats = [[] for user in userList]
    maxVal = 0

    # sort each list of times and generate iats from them
    if (not force) and os.path.isfile(DATAHOME+'iats.p'):
        print "iats file located. Loading..."
        iats = pickle.load(open(DATAHOME+'iats.p',"rb"))
        for user in iats:
            if len(user) > 0:
                maxOfUser = max(user)
                if maxVal < maxOfUser:
                    maxVal = maxOfUser

    else:
        for userNum, user in enumerate(userData):
            user = sorted(user)
            for i, time in enumerate(user):
                # Ignore where the user's first post (no preceeding time)
                if i == 0:
                    continue

                iat = time-user[i-1]
                if iat > maxVal:
                    maxVal = iat
                iats[userNum].append(iat)
        print "Finished calculating the interarrival times"
        pickle.dump(iats, open(DATAHOME+'iats.p',"wb"))

    return iats, maxVal

def generate_rts(bidTuples, userList, force=False):
    """
    Calculate the response times for each user
    Get the tuples in sorted order by user and organize into a new array by userNum

    :return: response times for each user, in order of userList
    """

    # bidTuples: [(userNum, "auctionId", timestamp)]
    # Assume train_seq_bidder_id_time.csv is sorted by user
    # bidTuples, userList = load_users(force)

    print "Finished adding nums to tuples. Sorting by auction........."

    # Sort by auction
    bidTuples = sorted(bidTuples, key=operator.itemgetter(1))

    # auctionData: [[(userNum, time),(userNum, time),(userNum, time)],[(userNum, time),(userNum, time),(userNum, time)]]
    # auctionList: ["auctionId0", "auctionId1"]
    if (not force) and os.path.isfile(DATAHOME+'auctionList.p') and os.path.isfile(DATAHOME+'auctionData.p'):
        print "auctionList file located. Loading..."
        auctionList = pickle.load(open(DATAHOME+'auctionList.p', "rb"))
        print "auctionData file located. Loading..."
        auctionData = pickle.load(open(DATAHOME+'auctionData.p', "rb"))
    else:
        print "auctionList or auctionData file not found. Generating."
        print "Finished sorting by auction. Grouping auctions together..."
        prevAuction = ""
        auctionList = []
        auctionData = []
        rowCount = 0
        for toks in bidTuples:
            userNum, auction, time = [toks[0], toks[1], toks[2]]
            # If this is a new auction, add to the list and data
            if not auction == prevAuction:
                auctionList.append(auction)
                auctionData.append([])

            # add the bid data to the last user on the list
            auctionData[-1].append((userNum, time))

            rowCount += 1
            if rowCount % 1000000 == 0:
                print rowCount
            prevAuction = auction
        print "Saving auctionList to file..."
        pickle.dump(auctionList, open(DATAHOME+'auctionList.p', "wb"))
        pickle.dump(auctionData, open(DATAHOME+'auctionData.p', "wb"))

    print "There are " + str(len(userList)) + " users."
    print "There are " + str(len(auctionData)) + " auctions."
    print "Calculating response times for auctions..."
    maxVal = 0

    # Go through each auction and sort by time
    # user[i] respondend in time[i]-time[i-1]
    if (not force) and os.path.isfile(DATAHOME+"rts.p"):
        print "rts file located. Loading..."
        rts = pickle.load(open(DATAHOME+'rts.p', "rb"))
        for user in rts:
            if len(user) > 0:
                maxOfUser = max(user)
                if maxVal < maxOfUser:
                    maxVal = maxOfUser
    else:
        rts = [[] for user in userList]
        for auction in auctionData:
            # Sorted by bid times
            # auction: [(user, time), (user, time), (user, time), ...]
            auction = sorted(auction, key=operator.itemgetter(1))
            for i, (userNum, time) in enumerate(auction):
                # user[i] respondend in time[i]-time[i-1]

                # Ignore where the user is the first to bid
                if i == 0:
                    continue

                rt = time-auction[i-1][1]
                if rt > maxVal:
                    maxVal = rt
                rts[userNum].append(rt)
        print "Finished calculating the response times"
        pickle.dump(rts, open(DATAHOME+"rts.p","wb"))
    return rts, auctionList, maxVal


def generate_bid_counts(bidTuples, userList, force=False):
    """
    Generate the number of bids total by user
    Generate the average number of bids per auction by user
    :data:
    :return:
    """
    if (not force) and os.path.isfile(DATAHOME+'userAuctionData.p'):
        print "userAuctionData file located. Loading..."
        userAuctionData = pickle.load(open(DATAHOME+'userAuctionData.p',"rb"))
    else:
        userAuctionData = [[] for user in userList]
        for toks in bidTuples:
            userNum, auction = [toks[0], toks[1]]
            userAuctionData[userNum].append(auction)
        pickle.dump(userAuctionData, open(DATAHOME+'userAuctionData.p', "wb"))

    userBidCounts = [0 for user in userList]
    userBidCountsPerAuction = [0 for user in userList]

    # userAuctionData contains lists of auctions that each user bid in (with duplicates)
    for userNum, user in enumerate(userAuctionData):
        userBidCounts[userNum] = len(user)
        userBidCountsPerAuction[userNum] = len(user)/len(set(user))

    return userBidCounts, userBidCountsPerAuction


def generate_num_devices(bidTuples, userList, force=False):
    # userDeviceCounts
    return [len(g) for g in group_unique_userdata(bidTuples, userList, BT_DEVICE, force=force)]


def generate_num_ips(bidTuples, userList, force=False):
    # userIPCounts
    return [len(g) for g in group_unique_userdata(bidTuples,userList,BT_IP, force=force)]


def generate_device_entropy(bidTuples, userList, force=False):
    return group_entropy(bidTuples, userList, BT_DEVICE, force=force)


def generate_IP_entropy(bidTuples, userList, force=False):
    return group_entropy(bidTuples, userList, BT_IP, force=force)


#########################
#                       #
#     H E L P E R S     #
#                       #
#########################

def load_users(force=False):
    """
    Stores the csv data in a list of tuples and replaces userId with a userNum, as per the generated userList
    Also returns a list of bots
    :param force:
    :return:
    """
    if (not force) and os.path.isfile(DATAHOME+'userList.p') and os.path.isfile(DATAHOME+'botList.p'):
        # print "userList file located. Loading..."
        userList = pickle.load(open(DATAHOME+'userList.p', "rb"))
        # print "botList file located. Locating..."
        botList = pickle.load(open(DATAHOME+'botList.p',"rb"))
    else:
        print "bidTuples file not found. Generating."
        print "Adding user nums to tuples..."
        userList = []
        userNum = -1
        prevUser = ""
        botList = []
        csvfile = open(DATAHOME+'train_seq_bidder_id_time.csv', 'rb')
        train_tbl = csv.reader(csvfile, delimiter=',', escapechar='\\', quotechar=None)
        # The csvfile should be sorted by username
        for toks in train_tbl:
            if not toks[0] == prevUser:
                userList.append(toks[USERID])
                userNum += 1
            if float(toks[3]) == 1:
                botList.append(userNum)
            prevUser = toks[0]

        print "Saving userList to file..."
        pickle.dump(userList, open(DATAHOME+'userList.p', "wb"))
        print "Saving botList to file..."
        pickle.dump(botList, open(DATAHOME+'botList.p', "wb"))

    botList = list(set(botList))
    return userList, botList


def load_data(force=False):
    """
    Stores the csv data in a list of tuples and replaces userId with a userNum, as per the generated userList
    Also returns a list of bots
    :param force:
    :return:
    """
    if (not force) and os.path.isfile(DATAHOME+'bidTuples.p') and os.path.isfile(DATAHOME+'userList.p') and os.path.isfile(DATAHOME+'botList.p'):
        print "bidTuples file located. Loading..."
        bidTuples = pickle.load(open(DATAHOME+'bidTuples.p', "rb"))
        print "userList file located. Loading..."
        userList = pickle.load(open(DATAHOME+'userList.p', "rb"))
        print "botList file located. Locating..."
        botList = pickle.load(open(DATAHOME+'botList.p',"rb"))
    else:
        print "bidTuples file not found. Generating."
        print "Adding user nums to tuples..."
        userList = []
        userNum = -1
        bidTuples = []
        prevUser = ""
        botList = []
        csvfile = open(DATAHOME+'train_seq_bidder_id_time.csv', 'rb')
        train_tbl = csv.reader(csvfile, delimiter=',', escapechar='\\', quotechar=None)
        # The csvfile should be sorted by username
        for toks in train_tbl:
            if not toks[0] == prevUser:
                userList.append(toks[USERID])
                userNum += 1
            bidTuples.append((userNum, toks[AUCTIONID], int(toks[TIMESTAMP]), toks[CATEGORY], toks[DEVICE], toks[IP]))

            if float(toks[3]) == 1:
                botList.append(userNum)
            prevUser = toks[0]

        print "Saving bidTuples to file..."
        pickle.dump(bidTuples, open(DATAHOME+'bidTuples.p', "wb"))
        print "Saving userList to file..."
        pickle.dump(userList, open(DATAHOME+'userList.p', "wb"))
        print "Saving botList to file..."
        pickle.dump(botList, open(DATAHOME+'botList.p', "wb"))

    botList = list(set(botList))
    return bidTuples, userList, botList


def group_unique_userdata(bidTuples, userList, columnNum, force=False):
    """
    Organizes data by user, keeping only unique values
    :param bidTuples:
    :param userList:
    :param columnNum:
    :param force:
    :return: row r contains the unique elements of columnNum for user r
    """
    userData = [[] for user in userList]

    # Sort by columnNum
    bidTuples = sorted(bidTuples, key=operator.itemgetter(columnNum))
    for toks in bidTuples:
        userNum = toks[0]
        userData[userNum].append(toks[columnNum])

    uniqueData = [[] for user in userList]
    for userNum, user in enumerate(userData):
        prevVal = None
        user = sorted(user, key=operator.itemgetter(columnNum))
        for col in user:
            if not col == prevVal:
                uniqueData[userNum].append(col)
            prevVal = col
    return uniqueData


def group_entropy(bidTuples, userList, columnNum, force=False):
    """
    Given a column number in bidTuples, calculate the average entropy for that column for each user
    :param bidTuples:
    :param userList:
    :param columnNum:
    :param force:
    :return:
    """
    userData = [[] for user in userList]
    userEntropies = [0 for user in userList]
    bidTuples = sorted(bidTuples, key=operator.itemgetter(columnNum))
    for toks in bidTuples:
        userNum = toks[0]
        userData[userNum].append(toks[columnNum])
    for userNum, user in enumerate(userData):
        # Should already be sorted by columnum from above...
        binCounts = []
        user = sorted(user, key=operator.itemgetter(columnNum))
        prevVal = None
        for col in user:
            if not col == prevVal:
                binCounts.append(0)
            binCounts[-1] += 1
            prevVal = col
        totalCounts = sum(binCounts)
        entropies = [np.log(float(count)/totalCounts)*float(count)/totalCounts for count in binCounts]
        userEntropies[userNum] = -sum(entropies)
    return userEntropies


def bucketize(data, log_base, outfile=None, maxVal=None):
    """
    Take in data and bucketize them by log_base
    input: N x D --> output: N x log(d) where d = max(data)
    N: number of users
    D: variable amount of data for each user (e.g. iats, rts, etc.)

    :param data: 2-d list of data. Each row is the list of iats,rts,etc. for a given user
    :param log_base: log base for bucketization
    :return: [[b_user1,b_user1,b_user1],[b_user2,b_user2,b_user2],[b_user3,b_user3,b_user3]]
    """
    if maxVal is None:
        # If maxVal is not specified, search for it
        max_datapoint = max(reduce(lambda x,y:x+y, data))
    else:
        max_datapoint = maxVal

    if outfile is not None:
        out = open(outfile, 'w')

    # Largest log-bucket
    largestBucket = int(1+math.floor(math.log(1+max_datapoint, log_base)))
    userBuckets = []
    for userNum, user in enumerate(data):
        buckets = [0 for i in range(largestBucket)]
        for d in user:
            bucket = int(math.floor(math.log(1+d, log_base)))
            buckets[bucket] += 1
        userBuckets.append(buckets)

        if outfile is not None:
            print >> out, " ".join([str(bucket) for bucket in buckets])

    return userBuckets


def mean_by_user(data):
    """
    Takes in a list of lists and returns a list of the means of each row
    :param data:
    :return:
    """
    # TODO (ajw): what to do about empty lists? e.g. no IATs
    # right now, they're np.NaN, but I use an Imputer to replace them with the mean
    return [np.mean(user) if len(user) > 0 else 0 for user in data]


def median_by_user(data):
    return [np.median(user) if len(user) > 0 else 0 for user in data]

def sparse_botlist(botList, userList):
    """
    Takes in a list of userNums of bots and returns a binary representation of the labels
    0 is human, 1 is bot
    :param botlist:
    :param userList:
    :return:
    """
    sparseList = [0]*len(userList)
    for botNum in botList:
        sparseList[botNum] = 1
    return np.array(sparseList)

def normalize_data(data):
    """
    Normalizes each column of data so the max is 1
    Note: this should be done before running SVC or logistic regression
    :param data:
    :return:
    """
    # normalize will normalize a row. We want column
    return normalize(data.T, norm="max").T

##################################
#                                #
#        F E A T U R E S         #
#                                #
##################################

def stack_features(featureList, nameList=None, outfile=None):
    """
    Takes in a list of numeric features and stacks them into a numpy matrix
    Each list in featureList stores a single feature for all users
    :param nameList:
    :param featureList:
    :param outfile:
    :return:
    """
    data = np.array(featureList).T
    if outfile is not None:
        # Print to csv
        if nameList is not None:
            assert(len(nameList) == len(featureList))
            head = ', '.join(nameList)
            np.savetxt(outfile, data, delimiter=",", header=head)
        else:
            np.savetxt(outfile, data, delimiter=",")
    return data


def six_features(force=False):
    userList, botList = load_users(force=False)
    # bidTuples, userList, botList = load_data(force=False)
    if (not force) and os.path.isfile(DATAHOME+"six_features.p"):
        six_features = pickle.load(open(DATAHOME+"six_features.p", "rb"))
        return six_features, sparse_botlist(botList, userList)
    else:
        bidTuples, userList, botList = load_data(force=False)
        meanIats = generate_mean_iats(bidTuples, userList, force=False)
        meanRts = generate_mean_rts(bidTuples, userList, force=False)
        bids, bidsPerAuction = generate_bid_counts(bidTuples, userList, force=False)
        numDevices = generate_num_devices(bidTuples, userList, force=False)
        numIps = generate_num_ips(bidTuples, userList, force=False)
        six_features = stack_features([meanIats, meanRts, bids, bidsPerAuction, numDevices, numIps], outfile=DATAHOME+"six_features.csv")
        pickle.dump(six_features, open(DATAHOME+"six_features.p","wb"))
        return six_features, sparse_botlist(botList, userList)


def five_features(force=False):
    userList, botList = load_users(force=False)
    if (not force) and os.path.isfile(DATAHOME+"five_features.p"):
        five_features = pickle.load(open(DATAHOME+"five_features.p", "rb"))
        return five_features, sparse_botlist(botList, userList)
    else:
        bidTuples, userList, botList = load_data(force=False)
        meanIats = generate_mean_iats(bidTuples, userList, force=False)
        meanRts = generate_mean_rts(bidTuples, userList, force=False)
        bids, bidsPerAuction = generate_bid_counts(bidTuples, userList, force=False)
        numDevices = generate_num_devices(bidTuples, userList, force=False)
        five_features = stack_features([meanIats, meanRts, bids, bidsPerAuction, numDevices], outfile=DATAHOME+"five_features.csv")
        pickle.dump(five_features, open(DATAHOME+"five_features.p","wb"))
        return five_features, sparse_botlist(botList, userList)


def five_and_rts(force=False):
    userList, botList = load_users(force=False)
    if (not force) and os.path.isfile(DATAHOME+"five_and_rts.p"):
        five_and_rts = pickle.load(open(DATAHOME+"five_and_rts.p",'rb'))
    else:
        bidTuples, userList, botList = load_data(force=False)
        five_feat = five_features(force=False)[0]
        rts = generate_binned_rts(bidTuples, userList, force=False)
        five_and_rts = np.concatenate((five_feat, rts), axis=1)
        pickle.dump(five_and_rts, open(DATAHOME+"five_and_rts.p", "wb"))
    # six_feats[1] is the sparse botList
    return five_and_rts, sparse_botlist(botList, userList)


def new_features(force=False):
    userList, botList = load_users(force=False)
    # bidTuples, userList, botList = load_data(force=False)
    if (not force) and os.path.isfile(DATAHOME+"new_features.p"):
        new_features = pickle.load(open(DATAHOME+"new_features.p", "rb"))
        return new_features, sparse_botlist(botList, userList)
    else:
        bidTuples, userList, botList = load_data(force=False)
        meanIats = generate_mean_iats(bidTuples, userList, force=False)
        bids, bidsPerAuction = generate_bid_counts(bidTuples, userList, force=False)
        numDevices = generate_num_devices(bidTuples, userList, force=False)
        deviceEntropy = generate_device_entropy(bidTuples, userList, force=False)
        ipEntropy = generate_IP_entropy(bidTuples, userList, force=False)
        new_features = stack_features([meanIats, bids, bidsPerAuction, numDevices, deviceEntropy, ipEntropy], outfile=DATAHOME+"new_features.csv")
        pickle.dump(new_features, open(DATAHOME+"new_features.p","wb"))
        return new_features, sparse_botlist(botList, userList)


def binned_time_features(force=False):
    if (not force) and os.path.isfile(DATAHOME+"binned_time_features.p"):
        binned_time_features = pickle.load(open(DATAHOME+"binned_time_features.p", "rb"))
    else:
        bidTuples, userList, botList = load_data(force=False)
        iats = generate_binned_iats(bidTuples, userList,force=False)
        rts = generate_binned_rts(bidTuples, userList, force=False)
        binned_time_features = np.concatenate((np.matrix(iats), np.matrix(rts)), axis=1)
        pickle.dump(binned_time_features, open(DATAHOME+"binned_time_features.p","wb"))
    return binned_time_features


def six_and_time_features(force=False):
    userList, botList = load_users(force=False)
    if (not force) and os.path.isfile(DATAHOME+"six_and_time_features.p"):
        six_and_time_feat = pickle.load(open(DATAHOME+"six_and_time_features.p",'rb'))
    else:
        six_feat = six_features(force=False)[0]
        binned_time_feat = binned_time_features(force=False)
        six_and_time_feat = np.concatenate((six_feat, binned_time_feat), axis=1)
        pickle.dump(six_and_time_feat, open(DATAHOME+"six_and_time_features.p", "wb"))
    # six_feat[1] is the sparse botList
    return six_and_time_feat, sparse_botlist(botList, userList)


def main():
    return

if __name__ == '__main__':
    main()
