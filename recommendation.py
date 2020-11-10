from datetime import datetime
import pandas as pd
import numpy as np
import math
from scipy import sparse
from tabulate import tabulate
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import pickle

with open('clf_model.pkl', 'rb') as file:  
    clf = pickle.load(file)
    
ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv("movies.csv")



def round(var):
    if (var<=0.5):
        return 0
    elif (var<=1.0):
        return 1
    elif (var<=1.5):
        return 1.5
    elif (var<=2.0):
        return 2
    elif (var<=2.5):
        return 2.5
    elif (var<=3.0):
        return 3
    elif (var<=3.5):
        return 3.5
    elif (var<=4.0):
        return 4
    elif (var<=4.5):
        return 4.5
    else:
        return 5



def cos_sim (source,sample_genre):
    cos_sim_list = []
    target = sample_genre
    L1 = len(tgram(source))
    L2 = len(tgram(target))
    v = tgram(source)
    b = tgram(target)
    total = v|b
    L = len(total)
    C = (L1+L2) - L
    ans = C/(math.sqrt(L1)*math.sqrt(L2))*100
    cos_sim_list.append(ans)
    return cos_sim_list

def tgram(instring) :
    combined = ""
    str_arr = instring.split('|')
    for e in range(len(str_arr)):
        combined = combined + str_arr[e]
    liststr = []
    op = len(combined) - 1
    ops = list(combined)
    for p in range(op):
        liststr.append(ops[p]+ops[p+1])  
    return set(liststr)
overall_sparse = sparse.csr_matrix((ratings.rating.values,(ratings.userId.values,ratings.movieId.values)))

def new_movie(sample_userId,sample_movieId):
    sample_movie_title = movies.loc[sample_movieId, 'title']
    sample = list()
    sample.append(overall_sparse.sum()/overall_sparse.count_nonzero()) #appending global average rating
    try:
        sample_similar_users = cosine_similarity(overall_sparse[sample_userId], overall_sparse).ravel()
        sample_similar_users_indices = np.argsort(-sample_similar_users)[1:]
        sample_similar_users_ratings = overall_sparse[sample_similar_users_indices, sample_movieId].toarray().ravel()
        sample_top_similar_user_ratings = list(sample_similar_users_ratings[sample_similar_users_ratings != 0][:5])
        sample_top_similar_user_ratings.extend([overall_sparse.sum()/overall_sparse.count_nonzero()]*(5-len(sample_top_similar_user_ratings)))
        sample.extend(sample_top_similar_user_ratings)
  #Cold Start    
    except(IndexError, KeyError):
        global_average_overall = [overall_sparse.sum()/overall_sparse.count_nonzero()]*5
        sample.extend(global_average_overall)
    try:
        sample_similar_movies = cosine_similarity(overall_sparse[:,sample_movieId].T, overall_sparse.T).ravel()
        sample_similar_movies_indices = np.argsort(-sample_similar_movies)[1:]
        sample_similar_movies_ratings = overall_sparse[sample_userId, sample_similar_movies_indices].toarray().ravel()
        sample_top_similar_movie_ratings = list(sample_similar_movies_ratings[sample_similar_movies_ratings != 0][:5])
        sample_top_similar_movie_ratings.extend([overall_sparse.sum()/overall_sparse.count_nonzero()]*(5-len(sample_top_similar_movie_ratings)))
        sample.extend(sample_top_similar_movie_ratings)
  #Cold Start
    except(IndexError, KeyError):
        global_average_overall = [overall_sparse.sum()/overall_sparse.count_nonzero()]*5
        sample.extend(global_average_overall)
        
    global_average_overall = overall_sparse.sum()/overall_sparse.count_nonzero()
    sample.append(global_average_overall)
    global_average_overall = overall_sparse.sum()/overall_sparse.count_nonzero()
    sample.append(global_average_overall)
    sample_arr = np.array(sample) 
    sample_arr = sample_arr.reshape(1, -1)
    sample_df = pd.DataFrame(sample_arr, columns=["Global_Average","SUR1", "SUR2", "SUR3", "SUR4", "SUR5","SMR1", "SMR2", "SMR3", "SMR4", "SMR5","User_Average", "Movie_Average"])
    predicted_rating = clf.predict(sample_df)
    rate = round(predicted_rating)
    sample_similar_rated = ratings[ratings["rating"] == rate]
    sample_similar_rated_movieIds = sample_similar_rated["movieId"].unique()
    movies_new = movies.copy()
    final_frame = movies_new.loc[movies_new.index.intersection(sample_similar_rated_movieIds)]
    sample_genre = movies_new.loc[sample_movieId,'genres']
    final_frame["similarity"] = final_frame['genres'].apply(cos_sim,sample_genre=sample_genre)
    final_frame.sort_values(by = "similarity", ascending = False, inplace=True)
    if sample_movieId in final_frame.index.values :
        final_frame.drop([sample_movieId], inplace=True)
    return final_frame.head(10)
    