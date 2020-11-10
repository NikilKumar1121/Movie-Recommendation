from flask import Flask, request, render_template, session, redirect,flash,url_for
import numpy as np
import pandas as pd
import json
import requests
from recommendation import new_movie,round,cos_sim,tgram
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)

#overall_sparse = sparse.csr_matrix((ratings.rating.values,(ratings.userId.values,ratings.movieId.values)))
ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv("movies.csv")
movies.set_index('movieId',inplace=True)
top_movie = pd.read_csv('top_movie.csv')
movie_genres= top_movie.groupby('genres')


def create_similarity():
    data = pd.read_csv('main_data.csv')
    # creating a count matrix
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['comb'])
    # creating a similarity score matrix
    similarity = cosine_similarity(count_matrix)
    return data,similarity

def rcmd(m):
    try:
        data.head()
        similarity.shape
    except:
        data, similarity = create_similarity()
    if m not in data['movie_title'].unique():
        return('Sorry! The movie you requested is not in our database. Please check the spelling or try with some other movies')
    else:
        i = data.loc[data['movie_title']==m].index[0]
        lst = list(enumerate(similarity[i]))
        lst = sorted(lst, key = lambda x:x[1] ,reverse=True)
        lst = lst[1:11] # excluding first item since it is the requested movie itself
        l = []
        for i in range(len(lst)):
            a = lst[i][0]
            l.append(data['movie_title'][a])
        return l


@app.route('/')
def home():
    return render_template('home.html')
    
@app.route('/top_genre.html', methods=["POST", "GET"])
def html_table():
    Genre = request.form["Genre"]
    Top_20_movies = pd.DataFrame(movie_genres.get_group(Genre).nlargest(20,['Overall_Rating_Count','Global_Average',"Movie_Average"]))
    Top_20_movies.rename(columns={'Unnamed: 0':'Movie Id','genres':'Genre','title':'Title'},inplace=True)
    Top_20_movies.reset_index(drop=True, inplace=True)
    return render_template('top_genre.html',  tables=[Top_20_movies.to_html(classes='data',index="False")],titles=Top_20_movies.columns.values)

@app.route('/recommend.html', methods=["POST","GET"])
def recommendation():
    sample_userId= int(request.form["User"])
    sample_movieId = int(request.form["Movie"])
    sample_movie_title = movies.loc[sample_movieId, 'title']
    sample_genre = movies.loc[sample_movieId, 'genres']
    Top_10_recommendation = new_movie(sample_userId,sample_movieId)           
    recommend= pd.DataFrame(Top_10_recommendation)
    recommend= pd.DataFrame(recommend)
    recommend.reset_index(drop=True,inplace=True)
    return render_template('recommend.html',title=sample_movie_title,tables=[recommend.to_html(classes="data",index="False")],titles=recommend.columns.values)
@app.route('/omdb.html',methods=["POST","GET"])
def omdb():
    if request.method =='POST': 
        title = request.form['title']
        url = "http://www.omdbapi.com/?t=" +title+ "&apikey=31dd8783" 
        ret = requests.get(url)
        moviedict = json.loads(ret.text)
        try:
            Movie_Title = moviedict['Title']
            Released_Date = moviedict['Released']
            Genre = moviedict['Genre'].replace(',','|')
            Director = moviedict['Director']
            Writer = moviedict['Writer']
            Actors = moviedict['Actors']
            production = moviedict['Production']
            Plot = moviedict['Plot']
            Awards = moviedict['Awards']
            Poster_url = moviedict['Poster']
            Imdb_Ratings = moviedict['imdbRating']
            Imdb_votes = moviedict['imdbVotes']
            movie_cards = recommend_posters(Movie_Title)
            return render_template('omdb.html',Movie_Title=Movie_Title,Released_Date=Released_Date,Genre=Genre,Director=Director,Writer=Writer,
                Actors=Actors,production=production,Imdb_votes=Imdb_votes,Plot=Plot,Awards=Awards,Poster_url=Poster_url,Imdb_Ratings=Imdb_Ratings,movie_cards=movie_cards)
        except:
            error= 'Sorry! The movie you requested is not in our database. Please check the spelling or try with some other movies'
            return render_template('home.html',error=error)


def recommend_posters(Movie_Title):
    m= Movie_Title
    m= m.lower()
    data, similarity = create_similarity()
    if m not in data['movie_title'].unique():
        error = "Sorry! The movie you requested Doesn't have recommendations"
        return error
    else:
        recommendations = rcmd(m)
        posters=[]
        Titles=[]
        for movie in recommendations:
            url = "http://www.omdbapi.com/?t=" + movie + "&apikey=31dd8783" 
            ret = requests.get(url)
            moviedict = json.loads(ret.text)
            Titles.append(moviedict['Title'])
            posters.append(moviedict['Poster'])
        movie_cards = {Titles[i]: posters[i] for i in range(len(posters))}
        return movie_cards


if __name__ == '__main__':
    app.run()