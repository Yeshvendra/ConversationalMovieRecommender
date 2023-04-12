################################################################
# File Description:
# The code to create recommendation based on the movie similarity is present in this file.
################################################################

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

df = pd.read_csv("./MovieSummaries/FinalFeatures.csv")
df = df.drop(columns=['Unnamed: 0'])
processedMovieNameList = []
for row in df['Movie_Name']:
    processedMovieNameList.append(row.lower())
df['Movie_Name_Preprocessed'] = processedMovieNameList

def RecommendMovieSimilarityBasedMovies(movieName, top = 10):
    movieName = movieName.lower()

    inputMovie_df = df[df['Movie_Name_Preprocessed'] == movieName]
    inputMovie_df = inputMovie_df.drop(columns=['Movie_ID', 'Movie_Plot', 'Movie_Revenue_Category', 'Movie_Name', 'Movie_Name_Preprocessed'])
    df_copy = df.drop(columns=['Movie_ID', 'Movie_Plot', 'Movie_Revenue_Category', 'Movie_Name', 'Movie_Name_Preprocessed'])

    movieList = []
    distanceList = pairwise_distances(inputMovie_df, df_copy, metric='euclidean')
    ind = distanceList[0].argsort()[1:top+1]
    for i in ind:
        row = df.iloc[i]
        movieList.append(row['Movie_Name'])
    return movieList

#print(RecommendMovieSimilarityBasedMovies("Ghosts of Mars"))
