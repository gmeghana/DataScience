from __future__ import division
import math
from operator import itemgetter
import operator
import random
import numpy as np
import sklearn
from sklearn.cluster import KMeans
from scipy.special import gammaln, psi

#BIRDNEST source code from author (adjusted for project purposes)
# runs bayesian fraud detection; returns vector of suspiciousness
# (in same order as iat_arr)

# @profile
def fit_alpha(D):
    (m, S) = D.shape
    S = int(S)
    if m <= 1: return [1] * S
    alpha = np.array([3] * S)
    alpha_next = np.array([None] * S)
    row_tot = np.sum(D, axis=1)
    MAX_FIT_ITERS = 100
    for it in range(MAX_FIT_ITERS):
        alpha_sum = np.sum(alpha)
        for s in range(S):
            D_col = D[:, s]
            numer = np.sum(D_col / (D_col + (-1 + alpha[s]) * np.ones(m)))
            denom = np.sum(row_tot / (row_tot + (-1 + alpha_sum) * np.ones(m)))
            alpha_next[s] = alpha[s] * numer / denom
        if np.sum(np.abs(alpha - alpha_next)) < 0.01:
            print "fitting iteration stopped early at iteration", it
            break
        alpha = alpha_next.copy()
        print alpha
    return alpha + 1

# @profile
def lbeta(alpha):
    return sum(math.lgamma(a) for a in alpha) - math.lgamma(sum(alpha))

# @profile
def ldirichlet_pdf(theta, alpha):
    kernel = sum((a - 1) * math.log(t) for a, t in zip(alpha, theta))
    return kernel - lbeta(alpha)

# @profile
def ldirich_multi_pdf(z, alpha):
    npalpha = alpha + z
    return (gammaln(np.sum(alpha)) - gammaln(np.sum(npalpha)) +
            np.sum(gammaln(npalpha)) - np.sum(gammaln(alpha)))

# denomiator of KL divergence, i.e. E_P 1/log Q(x), where P and Q have Dirich
# params alpha, beta;
# http://bariskurt.com/kullback-leibler-divergence-between-two-dirichlet-and-beta-distributions/
# @profile
def kl_denom(alpha, beta):
    # psi_sum = psi(sum(alpha))
    # psi_diff = sum([beta[k] * (psi_sum - psi(alpha[k])) for k in range(len(beta))])
    psi_diff = np.sum(beta * (psi(sum(alpha)) - psi(alpha)))
    return -math.lgamma(np.sum(beta)) + np.sum([math.lgamma(x) for x in np.squeeze(beta)]) + psi_diff

def detect_bot(iat_arr, use_times, K):
    m = iat_arr.shape[0]
    iat_sums = iat_arr.sum(axis=1) + 0.1
    iat_normal = iat_arr / iat_sums[:, np.newaxis] #Normalize THE IAT data
    #Itialize cluster assignment with Kmeans
    est = KMeans(n_clusters=K)
    est.fit(iat_normal) #changed from user_normal that concatenated rating_arr with iat_arr
    z = np.array(est.labels_)
    #z = np.zeros(m) #could have inialized with all users in cluster 1

    pi = [None] * K
    zn = np.array([-1] * m) # z_next
    (m, S2) = iat_arr.shape
    alpha2 = np.array([[0] * S2 for _ in range(K)], dtype=float)

    NUM_FIT_ITERS = 100
    for it in range(NUM_FIT_ITERS):
        print "iteration ", it
        for k in range(K):
            #Find all users in cluster k
            cur_idx = np.array((z == k))
            #Take only those users' IAT distributions
            iat_sub = iat_arr[cur_idx, :]
            #calculate how many in cluster
            n_k = np.sum(cur_idx)
            #calculate pi - cluster prior
            pi[k] = n_k / m
            #take subsample of users in cluster k
            if n_k > 1000:
                sample_idx = np.array(random.sample(range(n_k), 1000))
            else:
                sample_idx = range(n_k)
            print "subset size is ", len(sample_idx)
            iat_sub = iat_sub[sample_idx, :]
            #fit Dirichlet hyperparameters to distributions
            alpha2[k,:] = fit_alpha(iat_sub)
        print " alpha2 = ", alpha2

        print "fitting points"
        #Reassign cluster assignments of users given the log likelihood (Dirichlet-Mulinomial distribution)
        for i in range(m):
            log_likes = [ldirich_multi_pdf(iat_arr[i, :], alpha2[k]) for k in range(K)]
            zn[i] = log_likes.index(max(log_likes))
        num_diff = sum(abs(zn - z))
        z = zn
        #Test if model converged
        if num_diff == 0:
            print "Outer iteration stopped early at iteration ", it
            break
    #calculate posterior distributions for users
    post_iat = np.array(iat_arr, dtype='float')
    for i in range(m):
        post_iat[i,:] += alpha2[z[i]]

    #normalize scores to sum
    susp2 = np.zeros(m)
    for i in range(m):
        if i % 100000 == 0: print i
        susp2[i] = max([kl_denom(post_iat[i,:], alpha2[k,:]) for k in range(K)])

    susp2n = susp2 / np.std(susp2)
    suspn = (susp2n if use_times else 0)
    return suspn