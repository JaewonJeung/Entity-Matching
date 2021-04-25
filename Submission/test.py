import pandas as pd
import recordlinkage as rl
from recordlinkage.index import Block
import numpy as np

traindf = pd.read_csv('train.csv')
del traindf['label']
train_set = set()
for pair in traindf.values:
    train_set.add((pair[0], pair[1]))

pred_pairs = np.array(train_set)
pred_df = pd.DataFrame(train_set, columns=["ltable_id", "rtable_id"])
pred_df.to_csv("output.csv", index=False)