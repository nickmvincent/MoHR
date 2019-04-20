#%%
import tensorflow as tf
import numpy as np
import argparse
from tqdm import tqdm

from sampler import WarpSampler
from model import MoHR

#%%
dataset = np.load('AutomotivePartitioned.npy')

[user_train, user_validation, user_test, Item, usernum, itemnum] = dataset
user_train
#%%
user_train.keys()
#%%
user_ids = list(user_train.keys())
strike = np.random.choice(user_ids, size=int(0.3*len(user_ids)))
added = 0
mapping = {}
new_user_train = {}
new_user_val = {}
new_user_test = {}
for key in user_train.keys():
    if key not in strike:
        new_user_train[added] = user_train[key]
        new_user_val[added] = user_validation[key]
        new_user_test[added] = user_test[key]
        mapping[key] = added
        added +=1

usernum = len(user_train)

#%%
mapping

#%%
poprec = 0.64626
baseline = 0.787206
x = 0.765542 * 0.7 + 0.3 * poprec
x

#%%
diff = baseline - x
norm_diff = diff / (baseline - poprec)
diff, norm_diff

#%%
(baseline - 0.765542) / (baseline - poprec)
