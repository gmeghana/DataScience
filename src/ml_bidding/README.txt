### README for the classic ML approach
To run our model, change DATAHOME in ml_features.py to the parent of the
train_seq_bidder_id_time.csv file and run ml_models.py. The default is ./data

## ml_models.py
	This file contains functions I use to sweep, compare, and train different models
    Running main() will run through samples of all of these functions.

    > run_combined(): runs a classifier combining our optimal classifiers
    > run_importance(): Fit a classifier user all data and plot importances
    > plot_ROCList(): plot an ROC curve for each classifier in clfList

    # Cross-validation
    > run_clfList(): run 100-fold cross-validation on each classifier
    > kfoldcvList(): run k-fold cross-validation on each classifier
    > cvList(): run one instance of cross-validation on each classifier

    # Parameter Sweeps
    > sweep_svm(): sweeps over SVMs
    > sweep_logreg(): sweeps over logistic regression models
    > select_bestList(): sweeps over n_estimators for various classifiers
    > select_bestList_higher(): sweeps over n_estimators for high values of n
    > select_best_combined(): sweeps over different combination models
    > select_bestListList(): sweeps over n_estimators for various classifiers,
                           : using the same training/test set for each

    > class CombinedClassifier

## ml_features.py
	This file contains the functions I use to preprocess some features from the
    training data. You should only have to call one of the complete feature-set
    generation functions, and the rest of the necessary files should either be
    found automatically or should be generated and saved for future use.

    To run correctly, set DATAHOME to be the path to the directory containing
    train_seq_bidder_id_time.csv.

	Some functions have a :force: input parameter. If this is true, the function
    will go through the entire computation. Otherwise, it will try to load load
    data from each variable's corresponding pickle file (forced if the file is
    not found in DATAHOME).

    # Complete Feature-set Generation
    > six_features():
        The original six features we tried
        [meanIats, meanRts, bids, bidsPerAuction, numDevices, numIps]
    > six_and_time_features():
        The original six features with the binned distributions
    > five_features():
        The original six features we tried, minus numIps
        [meanIats, meanRts, bids, bidsPerAuction, numDevices
    > five_and_rts():
        The five features and the bucketized responds times
    > binned_time_features():
        The binned time distributions
    > new_features():
        The final model
        [meanIats, bids, bidsPerAuction, numDevices, deviceEntropy, ipEntropy]

    # Generating individual features and files
    The following files have inputs (bidTuple, userList, force(optional))
    They all generate pickle files and data specified in their names
    The inputs bidTuple, userList are loaded from load_users() or load_data()
    Additional files may be created, as specified below each function.

    > generate_mean_iats()
    > generate_iats()
        userData: dict mapping userNum to list of bid times
    > generate_mean_rts()
    > generate_rts()
        auctionData: dict mapping auctionNum to list of (userNum, bidTime)
        auctionList: auctionList[auctionNum] = "auctionId string"
    > generate_bid_counts()
        userAuctionData: userAuctionData[userNum] = [user's list of auctions]
        userBidCounts: list of bid counts for user at index userNum
        userBidCountsPerAuction: list of mean bid counts per auction
    > generate_num_devices()
    > generate_num_ips()
    > generate_device_entropy()
    > generate_IP_entropy()

    # Helpers and Loading Data
    These are helper functions that get called by other functions in generating
    the features.

    >load_data()
        bidTuples: [(userNum, auctionId, time, category, device, IP)]
        botList: list of userNums that correspond to bots
        userLst: list of userIds
    > load_users()
        botList: " " " "
        userLst: " " " "
    > group_entropy()
    > group_unique_userdata()
    > bucketize()
    > normalize_data()
    > mean_by_user()
    > median_by_user()
    > sparse_botlist()
    


