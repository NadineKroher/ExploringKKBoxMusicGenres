import pandas as pd
import numpy as np
import itertools

# read songs into pandas data frame
songs = pd.read_csv('../data/songs.csv')
songs = songs[['song_id','genre_ids']] # we don't need the rest

# convert genre ids to list
songs['genre_ids'] = songs['genre_ids'].map(lambda x: [int(y) for y in str(x).split('|')] if not pd.isnull(x) else [])

# get unique list of genre ids
genres = songs['genre_ids'].values.tolist()
genres = [j for i in genres for j in i]
genres = list(set(genres))

# number of unique genres
numGenres = len(genres)

# init genre similarity matrix D and genre score list s
D = np.zeros((numGenres,numGenres))
s = np.zeros((numGenres,))

# read listening data
listen = pd.read_csv('../data/train.csv')

# we only consider songs with target 1
listen = listen[listen['target'] == 1]
listen = listen[['msno','song_id']]

# join the two datasets
songs.set_index('song_id', inplace=True)
df = listen.join(songs, how="left", on="song_id")
df.dropna(axis=0,inplace=True)

# group by user and process groups
for user, frame in df.groupby('msno'):
    userGenres = frame['genre_ids'].values.tolist()
    userGenres = list(itertools.chain.from_iterable(userGenres))
    userGenres = set(list(userGenres))
    combs = itertools.combinations(userGenres, 2)
    for comb in combs:
        print(comb)

print(df.head(5))
