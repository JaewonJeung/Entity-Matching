import pandas as pd
import recordlinkage as rl
from recordlinkage.index import Block
import numpy as np
import re

ltdf = pd.read_csv('ltable.csv')
rtdf = pd.read_csv('rtable.csv')
traindf = pd.read_csv('train.csv')

irrelevant_regex = re.compile(r'[^a-z0-9\s]')
multispace_regex = re.compile(r'\s\s+')


def assign_no_symbols_modelno(df):
    return df.assign(
        modelno=df['modelno']
            .str.replace(irrelevant_regex, '')
            .str.replace(multispace_regex, ''))


def assign_no_symbols_title(df):
    return df.assign(
        title=df['title']
            .str.replace(irrelevant_regex, '')
            .str.replace(multispace_regex, ''))


ltdf = assign_no_symbols_modelno(ltdf)
rtdf = assign_no_symbols_modelno(rtdf)
ltdf = assign_no_symbols_title(ltdf)
rtdf = assign_no_symbols_title(rtdf)

indexer = Block('brand')
pairs = indexer.index(ltdf, rtdf)

comp = rl.Compare()
comp.string('title', 'title', method='jarowinkler', label='title')
comp.string('category', 'category', method='jarowinkler', label='category')
comp.string('brand', 'brand', method='jarowinkler', label='brand')
comp.string('modelno', 'modelno', method='jarowinkler', label='modelno')
comp.numeric('price', 'price', label='price')

comparison_vectors = comp.compute(pairs, ltdf, rtdf)

scores = np.average(
    comparison_vectors.values,
    axis=1,
    weights=[50, 10, 10, 70, 20])
scored_comparison_vectors = comparison_vectors.assign(score=scores)

matches = scored_comparison_vectors[
    scored_comparison_vectors['score'] >= 0.80]
# print(matches.head(25))

found_pairs_set = set(matches.index)
# print(len(found_pairs_set))

del traindf['label']
train_set = set()
for pair in traindf.values:
    train_set.add((pair[0], pair[1]))

pred_pairs = found_pairs_set - train_set
# print(len(pred_pairs))

pred_df = pd.DataFrame(pred_pairs, columns=["ltable_id", "rtable_id"])
pred_df.to_csv("output.csv", index=False)
