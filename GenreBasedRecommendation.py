################################################################
# File Description:
# The code to create recommendation based on the genre is present in this file.
################################################################

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

df = pd.read_csv("./MovieSummaries/FinalFeatures.csv")
df = df.drop(columns=['Unnamed: 0'])

def RecommendGenreBasedMovies(genre, top = 10):
    genre = genre.lower()
    if genre in df.columns:
        genre_df = df[df[genre] == 1]
        final_df = genre_df[genre_df['Movie_Revenue_Category'] == 'High']
        row_count = final_df.shape[0]
        if row_count < 10:
            final_df = genre_df[genre_df['Movie_Revenue_Category'] == 'High' 
                                and genre_df['Movie_Revenue_Category'] == 'High_Med']
            row_count = final_df.shape[0]
            if row_count < 10:
                final_df = genre_df[genre_df['Movie_Revenue_Category'] == 'High' 
                                and genre_df['Movie_Revenue_Category'] == 'High_Med'
                                and genre_df['Movie_Revenue_Category'] == 'Low_Med']
                row_count = final_df.shape[0]
                if row_count < 10:
                    final_df = genre_df
        row_count = final_df.shape[0]
        if row_count < top:
            top = row_count
        final_df_copy = final_df.drop(columns=['Movie_ID', 'Movie_Plot', 'Movie_Revenue_Category', 'Movie_Name'])
        kmeans = KMeans(n_clusters=1, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(final_df_copy)

        movieList = []
        distanceList = pairwise_distances(kmeans.cluster_centers_, final_df_copy, metric='euclidean')
        ind = distanceList[0].argsort()[:top]
        for i in ind:
            row = final_df.iloc[i]
            movieList.append(row['Movie_Name'])
        return movieList
    else:
        return []

#print(RecommendGenreBasedMovies("action"))
