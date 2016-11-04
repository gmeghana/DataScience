__author__ = "andrew"

import random
import numpy as np
from ml_features import *
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_curve
from sklearn.preprocessing import Imputer
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import sys

def plot_ROCList(clfList, data, labels, stringList=""):
    """
    Plot an ROC curve for each classifier in clfList, training on a single 80/20 split
    :param clfList:
    :param data:
    :param labels:
    :param stringList:
    :return:
    """
    if stringList == "":
        stringList = ["" for i in range(len(labels))]
    imp = Imputer(missing_values=np.NaN, strategy="mean")
    data = imp.fit_transform(data)

    # Cross-validate on the data once using each model to get a ROC curve
    AUCs, fprs, tprs, threshs = cvList(data, labels, clfList)

    # Plote a ROC for each clf in clfList
    for i in range(len(clfList)):
        fpr = fprs[i]
        tpr = tprs[i]
        plt.plot(fpr, tpr)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(stringList[i]+" ROC Curve, AUC = "+str(AUCs[i]))
        plt.savefig(stringList[i]+"_ROC.png")
        plt.close()
        print stringList[i] + ":" + str(AUCs[i])


def run_clfList(clfList, stringList="", normalize=False):
    """
    Run 100-fold 80/20 cross-validation on each classifier in clfList
    print the average AUC for each classifier
    :param clfList: list of classifiers to run
    :param stringList: names of the classifiers
    :param normalize: whether or not to normalize the data
    :return: the average AUC for each classifier in clfList
    """
    # data, labels = six_features(force=False)
    # data, labels = six_and_time_features(force=False)
    # data, labels = five_features(force=False)
    # data, labels = five_and_rts(force=False)
    data, labels = new_features()
    if normalize:
        data = normalize_data(data)

    imp = Imputer(missing_values=np.NaN, strategy="mean")
    data = imp.fit_transform(data)

    # Cross-validate all clfs 100 times
    means = kfoldcvList(data, labels, clfList, 100)
    if stringList == "":
        stringList = ["" for i in range(len(labels))]

    # Print out the mean AUCs
    for i, mean in enumerate(means):
        print stringList[i]+": "+str(mean)

    for mean in means:
        sys.stdout.write(str(mean) + " & ")
    sys.stdout.write("\n")
    return means


def select_bestList_higher(clf, string="--"):
    """
    Selects the best value for n_estimators for clf, judging by cross-validated AUCs
    :param clf:
    :param string:
    :return:
    """
    # Train clf over a larger range of classifiers
    n_range = ([2, 10, 50, 100, 200, 300, 400, 500])
    clfList = [clf(n_estimators=n) for n in n_range]
    stringList = [string+" ("+str(n)+")" for n in n_range]
    AUCs = run_clfList(clfList, stringList)

    plt.title(string+ " AUC vs Number of Estimators")
    plt.xlabel("Number of Estimators")
    plt.ylabel("AUC")
    plt.scatter(n_range, AUCs)
    plt.savefig("AUCvN_highN_"+string+".png")
    plt.close()


def select_bestList(clf, string="--"):
    """
    Selects the best value for n_estimators for clf, judging by cross-validated AUCs
    :param clf:
    :param string:
    :return:
    """
    n_range = (range(2,11)+[15,30,50,100,150,200])
    # n_range = [3,5,7,9,15,30]
    clfList = [clf(n_estimators=n) for n in n_range]

    # To test classifier with different base_estimator, uncomment the following line (also recommended, change n_range)
    # clfList += [clf(n_estimators=n, base_estimator=SVC(kernel="linear", C=50, probability=True)) for n in n_range]
    stringList = [string+" ("+str(n)+")" for n in n_range]
    AUCs = run_clfList(clfList, stringList)
    plt.title(string+ " AUC vs Number of Estimators")
    plt.xlabel("Number of Estimators")
    plt.ylabel("AUC")
    plt.scatter(n_range, AUCs)
    plt.savefig("AUCvN_"+string+".png")
    plt.close()


def select_bestListList(clfList, stringList=""):
    """
    Run multiple classifiers on the same cross-validated training set and select the best value for each clf
    :param clfList:
    :param stringList:
    :return:
    """
    n_range = (range(2,11)+[15,30,50,100,150,200])
    clfListList = []
    stringListList = []

    # Merge the list of classifier classes into actual classifiers ranging over n and then train
    for clf, string in zip(clfList, stringList):
        clfListList += [clf(n_estimators=n) for n in n_range]
        stringListList += [string+" ("+str(n)+")" for n in n_range]

    run_clfList(clfListList, stringListList)


def select_best_combined(n):
    """
    Combine multiple of the same classifier and select the best number of estimators
    Note: In order to change the classifier, need to change them inline.
    :param n: The number of estimators in each boosted model
    :return: (void) print the AUCs for each combined classifier
    """
    Ks = [1,3,5,7,9]
    combs = []
    for k in Ks:
        # Set up default classifiers
        comblist = [AdaBoostClassifier() for i in range(k)]

        # Create k copies of each clf and combine
        for i in range(k):
            comblist[i] = AdaBoostClassifier(n_estimators=n, random_state=i)
        combs.append(CombinedClassifier(comblist))
    run_clfList(combs, ["1", "3", "5", "7", "9"])


def run_combined():
    """
    Runs the classifier combining all five of our optimal classifiers
    Also runs each individual classifier for comparison
    :return:
    """
    grad1 = AdaBoostClassifier(n_estimators=8)
    grad2 = GradientBoostingClassifier(n_estimators=30)
    grad3 = RandomForestClassifier(n_estimators=200)
    grad4 = BaggingClassifier(n_estimators=200)
    grad5 = ExtraTreesClassifier(n_estimators=200)
    comb = CombinedClassifier([grad1, grad2, grad3, grad4, grad5])
    grad1 = AdaBoostClassifier(n_estimators=8)
    grad2 = GradientBoostingClassifier(n_estimators=30)
    grad3 = RandomForestClassifier(n_estimators=200)
    grad4 = BaggingClassifier(n_estimators=200)
    grad5 = ExtraTreesClassifier(n_estimators=200)
    run_clfList([comb, grad1, grad2, grad3, grad4, grad5], ["Combined", "Adaboost", "Gradient Boosting", "Random Forest", "Bagging", "Extra Trees"])


def run_importance(clf, data, labels, feature_labels=[""], string=""):
    """
    Fit a classifier using all the data and plot the feature importances
    :param clf: Classifier object that has feature_importances_ member
    :param feature_labels: names of the features
    :param string: classifier name
    :return: (void) plot Gini importance vs feature
    """
    num_features = data.shape[1]
    importances = [0]*num_features

    imp = Imputer(missing_values=np.NaN, strategy="mean")
    data = imp.fit_transform(data)

    # run the classifier 100 times and average the importance found after each fit
    for r in range(100):
        clf.fit(data, labels)
        importances = [importances[i]+clf.feature_importances_[i] for i in range(num_features)]
    importances = [importance/100 for importance in importances]

    # Filter out the features that have 0 importance (e.g. values are all 0)
    # non_zeros are the indices in feature_importances that are not 0
    non_zeros = [i for i in range(num_features) if not importances[i] == 0]
    importances = [importances[i] for i in non_zeros]
    feature_labels = [feature_labels[i] for i in non_zeros]

    # Plot the features
    bar_width = 0.7
    plt.bar(range(len(feature_labels)), importances, bar_width)
    plt.xticks([ind + +float(bar_width)/2 for ind in range(len(feature_labels))], feature_labels,rotation="vertical")
    plt.gcf().subplots_adjust(bottom=0.35)
    plt.xlabel("Feature")
    plt.ylabel("Gini Importance")
    plt.title("Gini Importance v. Features for "+string+" Classifier")
    plt.show()


def kfoldcvList(data, labels, clfList, k):
    """
    Apply k-fold cross-validation with 80/20 split
    :param data: features for each user
    :param labels: bot/human labels
    :param clfList: list of classifiers to try training
    :param k: number of different folds
    :return: (void) plot Gini importance vs feature
    """
    AUCs = [[] for i in range(k)]
    for i in range(k):
        # cvList is a single instance of 80/20 split training/testing for all classifiers in clfList
        # It returns the AUC for that, the false/true positive rates, and thresholds for ROC
        AUCs[i], fprs, tprs, threshs = cvList(data, labels, clfList)
        sys.stdout.write('|')
    sys.stdout.write("\n")

    # AUCs is k x num_models
    # axis=0 makes this 1 x num_models
    return np.mean(np.array(AUCs), axis=0)


def cvList(data, labels, clfList):
    """
    Split the data into 80/20 split once and test the classifiers in clfList on them
    :param data:
    :param labels:
    :param clfList: list of classifiers to train/test on
    :return: AUC scores for each classifier in clfList
    """
    # Keep shuffling the data until we arrive at a partition that has at least 20 bots in each training/test set
    while True:
        shuffled = range(len(labels))
        random.shuffle(shuffled)
        twentyMark = len(labels)/5
        test_data = data[shuffled[:twentyMark],:]
        test_labels = labels[shuffled[:twentyMark]]
        train_data = data[shuffled[twentyMark:],:]
        train_labels = labels[shuffled[twentyMark:]]
        if sum(train_labels) > 20:
            break

    # Get the fpr, tpr, thresh for each and the AUCs for each
    aucList = [0]*len(clfList)
    tprList = [[] for i in range(len(clfList))]
    fprList = [[] for i in range(len(clfList))]
    threshList = [[] for i in range(len(clfList))]
    for i,clf in enumerate(clfList):
        clf.fit(train_data, train_labels)
        estimate_scores = clf.predict_proba(test_data)
        aucList[i] = roc_auc_score(test_labels,estimate_scores[:,1])
        fpr, tpr, thresholds = roc_curve(test_labels, estimate_scores[:,1], pos_label=1)
        tprList[i] = tpr
        fprList[i] = fpr
        threshList[i] = thresholds

    return aucList, fprList, tprList, threshList


def sweep_svm():
    """
    Sweep through different kernels and Cs for SVMs
    :return: (void) print cross-validated AUCs for each
    """
    Cs = [0.1, 0.5, 1, 5, 20, 50, 100, 150, 200, 300]
    kernels = ['linear', 'poly', 'rbf']
    polys = [2,3,4]

    stringList = []
    svmList = []
    for kern in kernels:
        for C in Cs:
            if kern == 'poly':
                for deg in polys:
                    svmList.append(SVC(C=C, kernel=kern, degree=deg, probability=True))
                    stringList.append(kern + " deg"+str(deg)+ " C(" + str(C)+")")
            else:
                svmList.append(SVC(C=C, kernel=kern, probability=True))
                stringList.append(kern + " C(" + str(C)+")")
    run_clfList(svmList, stringList,normalize=True)


def sweep_logreg():
    """
    Sweeps over a range of values for C, and different optimizations (primal and dual
    :return: (void) print cross-validated AUCs for each
    """
    Cs = [0.1, 0.5, 1, 5, 20, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1500, 2000, 3000, 4500, 7500, 10000, 15000, 20000]
    logregList = []
    stringList = []
    for C in Cs:
        logregList.append(LogisticRegression(dual=False, C=C))
        stringList.append(str(C) + " primal")
        logregList.append(LogisticRegression(dual=True, C=C))
        stringList.append(str(C) + " dual")
    run_clfList(logregList, stringList, normalize=True)


class CombinedClassifier:
    """
    Classifier that trains multiple classifiers and takes the median of their output as result
    """
    def __init__(self, clfList):
        print "New combined classifier:"
        for clf in clfList:
            print id(clf)
        self.clfList = clfList

    def fit(self, data, labels):
        for clf in self.clfList:
            clf.fit(data, labels)

    def predict_proba(self, test_data):
        probas = []
        for clf in self.clfList:
            probas.append(clf.predict_proba(test_data))
        # print  np.median(probas, axis=0)
        return np.median(probas, axis=0)



def main():
    # data are the features
    # labels are the binary labels for human/bot
    data, labels = six_features(force=False)
    data, labels = six_and_time_features(force=False)
    data, labels = five_features(force=False)
    data, labels = five_and_rts(force=False)
    data, labels = new_features(force=False)

    # plot the Gini importance values for
    run_importance(GradientBoostingClassifier(n_estimators=50), data, labels, ["meanIats", "bids", "bidsPerAuction", "numDevices", "deviceEntropy", "ipEntropy"], "GradientBoost")

    # Plot the ROC for a single 80/20 partition
    plot_ROCList([GradientBoostingClassifier(n_estimators=50)], data, labels, ["GradientBoost"])

    ####################################################################################################
    # ALL OF THE CODE BLOW WILL RUN USING new_features() DATA. To change, change line in run_clfList() #
    ####################################################################################################

    # Sweep over C for SVC, and over kernels (linear, poly[2,3,4], rbf)
    sweep_svm()

    # Sweep over C for LogisticRegression, and optimization primal v. dual
    sweep_logreg()

    # Sweep over n_estimators for various classifier types individually
    select_bestList(AdaBoostClassifier, "Adaboost")
    select_bestList(GradientBoostingClassifier, "Gradient Boosting")
    select_bestList(RandomForestClassifier, "Random Forest")
    select_bestList(BaggingClassifier, "Bagging Classifier")
    select_bestList(ExtraTreesClassifier,"Extra Trees")

    # Sweep over n estimators for various classifiers using the same training/test set
    select_bestListList([AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier], ["Adaboost", "Gradient Boosting", "Random Forest", "Bagging Classifier","Extra Trees"])

    # Sweep over a higher range of n_estimators for select classifiers
    select_bestList_higher(RandomForestClassifier, "Random Forest")
    select_bestList_higher(BaggingClassifier, "Bagging Classifier")
    select_bestList_higher(ExtraTreesClassifier,"Extra Trees")

    # Try multiple of a single AdaboostClassifier, n_estimators=10
    # To change to a different classifier, modify select_best_combined()
    select_best_combined(10)

    # Try a combination of a the above classifiers, with pre-selected optimal n_estimators
    run_combined()


if __name__ == "__main__":
    main()

####################################
#                                  #
#   D E P R E C A T E D  C O D E   #
#                                  #
####################################
#
# def run_clf(clf, string=""):
#     data, labels = new_features()
#     # data, labels = five_and_rts()
#     # data, labels = six_and_time_features()
#
#     imp = Imputer(missing_values=np.NaN, strategy="mean")
#     data = imp.fit_transform(data)
#     mean, AUC, tpr, fpr, threshs = kfoldcv(data, labels, clf, 100)
#     # tpr = np.matrix(tpr)
#     # fpr = np.matrix(fpr)
#     # threshs = np.array(threshs)
#     print string+":", str(mean)
#     # print AUC
#     # print tpr
#     return mean, tpr, fpr
#
#
# def select_best(clf, string="--"):
#     n_range = (range(2,11)+[15,30,50,100,150,200])
#     AUCs = []
#     for n in n_range:
#         meanAUC, tpr, fpr = run_clf(clf(n_estimators=n),string+" ("+str(n)+")")
#         AUCs.append(meanAUC)
#     sys.stdout.write("\n")
#     plt.scatter(n_range,AUCs)
#     plt.xlabel("n estimators")
#     plt.ylabel("AUC")
#     plt.title(string+" AUC vs. Number of Estimators")
#     plt.savefig(string+"_AUC.png")
#     plt.close()
#     print string+" best n:", str(n_range[AUCs.index(max(AUCs))])
#     return n_range[AUCs.index(max(AUCs))]
# def run_decisiontree():
#     data, labels = six_and_time_features()
#     data = binned_time_features()
#     # data, labels = five_and_rts()
#     num_features = data.shape[1]
#     print data.shape
#     importances = [0]*num_features
#     # dtree = DecisionTreeClassifier()
#     dtree = GradientBoostingClassifier(n_estimators=30)
#     imp = Imputer(missing_values=np.NaN, strategy="mean")
#     data = imp.fit_transform(data)
#
#     for r in range(100):
#         dtree.fit(data, labels)
#         importances = [importances[i]+dtree.feature_importances_[i] for i in range(num_features)]
#     importances = [importance/100 for importance in importances]
#
#     six_labels = []#["meanIats", "meanRts", "bids", "bidsPerAuction", "numDevices", "numIps"]
#     # six_labels = ["meanIats", "meanRts", "bids", "bidsPerAuction", "numDevices"]
#     iat_labels = ["IAT bin "+str(k) for k in range(47)]
#     rt_labels = ["RT bin "+str(k) for k in range(47)]
#     all_labels = six_labels+iat_labels+rt_labels
#     # all_labels = six_labels + rt_labels
#
#
#
#     dtreeMean, dtreeAUC, dtreeTpr, dtreeFpr,dtreeThresh = kfoldcv(data, labels, dtree, 100)
#     print dtreeMean
#     print dtreeAUC
#     print dtree.feature_importances_
#
#
# def kfoldcv(data, labels, clf, k):
#     AUCs = [0]*k
#     ROC_tpr = [[] for i in range(k)]
#     ROC_fpr = [[] for i in range(k)]
#     ROC_thresh = [[] for i in range(k)]
#     for i in range(k):
#         # sys.stdout.write("|")
#         AUCs[i], ROC_fpr[i], ROC_tpr[i], ROC_thresh[i] = cv(data, labels, clf)
#     # sys.stdout.write("\n")
#
#     return np.mean(np.array(AUCs)), np.array(AUCs), ROC_tpr, ROC_fpr, ROC_thresh
#
#
# def cv(data, labels, clf):
#     """
#     Splits the data 80/20 and tests with AUC score and ROC curve
#     :param data:
#     :param labels:
#     :param clf:
#     :return:
#     """
#     # data and labels are both np.array type
#     while True:
#         shuffled = range(len(labels))
#         random.shuffle(shuffled)
#         twentyMark = len(labels)/5
#         test_data = data[shuffled[:twentyMark],:]
#         test_labels = labels[shuffled[:twentyMark]]
#         train_data = data[shuffled[twentyMark:],:]
#         train_labels = labels[shuffled[twentyMark:]]
#         if sum(train_labels) > 20:
#             break
#     clf.fit(train_data, train_labels)
#     estimate_scores = clf.predict_proba(test_data)
#     fpr, tpr, thresholds = roc_curve(test_labels, estimate_scores[:,1], pos_label=1)
#     return roc_auc_score(test_labels, estimate_scores[:,1]), fpr, tpr, thresholds
