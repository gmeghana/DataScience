__author__ = 'MsSaraMel'

import numpy as np
import csv
import sys
import pandas as pd

#with open('../Data/bids.csv', 'r') as f:
#    reader = csv.reader(f);
#    data = [data for data in reader];
#data_array = np.asarray(data);

def import_data():

    bidder = pd.read_csv("../Data/train.csv")
    bids = pd.read_csv("../Data/bids.csv")
    bids = bids.dropna(axis=1)
    merged = bidder.merge(bids, on='bidder_id')
    merged = merged.sort(['bidder_id', 'time'])
    merged.to_csv("../Data/sorted_join_train.csv", index=False)


    print "finished"

def merge_dt_ordered_join():

    left_table = pd.read_csv('../Data/train_ora_dedup.csv', header=0)
    left_table.rename(columns = {'BIDDER_ID': 'bidder_id'})
    left_table.rename(columns = {'TIME': 'time'})
    left_table.rename(columns = {'AUCTION': 'auction'})
    left_table.rename(columns = {'OUTCOME': 'outcome'})
    right_table = pd.read_csv('../Data/Response_Time_Table.csv', header=0)
    key = ['bidder_id', 'auction', 'time']
    merged = left_table.merge(right_table, how='left', on=key)
    merged = merged.sort(['bidder_id', 'time'])
    merged.to_csv("../Data/sorted_BIG_train.csv", index=False)

    print "finished"

def get_rid_of_white_spaces():
    with open('../Data/train_iat_rsp.csv','rw') as file:
        for line in file:
            if line.strip():
                file.write(line)

if __name__ == '__main__':
    #merge_dt_ordered_join()
    get_rid_of_white_spaces()