import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import hashlib

HOUSING_PATH = os.path.join("datasets","housing")

def load_housing_data(): #housing_path=HOUSING_PATH
    #csv_path = os.path.join(housing_path,"housing.csv")
    return pd.read_csv("housing.csv")

housing = load_housing_data()
housing.head()

housing.hist(bins=50,figsize=(20,15))
plt.show()

#Creating test set based on reshuffle of indexes
def split_train_test(data,test_ratio):
    shuffled_idx = np.random.permutation(len(data)) #It deorganises the data
    test_set_size = int(len(data)*test_ratio)
    test_indx = shuffled_idx[:test_set_size]
    train_idx = shuffled_idx[test_set_size:]
    return data.iloc[train_idx], data.iloc[test_indx] #It rebuilds the data based on the indexes we got from the randomization

#Creating test set based on hash calculation
def test_set_check(identifier,test_ratio,hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio

def split_train_test_hash(data,test_ratio,id_column,hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_,test_ratio,hash))
    return data.loc[~in_test_set], data.loc[in_test_set]

housing_withID = housing.reset_index() #adds index column to data set to test split approach with hash
train_set,test_set = split_train_test_hash(housing_withID,0.2,"index")

from sklearn.model_selection import train_test_split
train_set_sk,test_set_sk = train_test_split(housing,test_size=0.2,random_state=42)

housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)