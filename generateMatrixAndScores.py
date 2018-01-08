import pandas as pd
import numpy as np
import itertools
import pickle

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
    print(user)
    userGenres = frame['genre_ids'].values.tolist()
    userGenres = list(itertools.chain.from_iterable(userGenres))
    userGenres = set(list(userGenres))

    for aGenre in userGenres:
        m = genres.index(aGenre)
        s[m] += 1

    combs = itertools.combinations(userGenres, 2)
    for comb in combs:
        D[genres.index(comb[0]),genres.index(comb[1])] += 1
        D[genres.index(comb[1]),genres.index(comb[0])] += 1

# save everything as a pickle file
pickle.dump(D,open('D.p','wb'))
pickle.dump(genres,open('genres.p','wb'))
pickle.dump(s,open('s.p','wb'))
