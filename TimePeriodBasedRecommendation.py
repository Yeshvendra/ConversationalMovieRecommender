################################################################
# File Description:
# The code to create recommendation based on the given time period is present in this file.
################################################################

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

df = pd.read_csv("./MovieSummaries/FinalFeatures.csv")
df = df.drop(columns=['Unnamed: 0'])

def RecommendTimePeriodBasedMovies(start_year, end_year, top = 10):
    start_year = int(start_year)
    end_year = int(end_year)
    
    time_df = df[df['Movie_Release_Year'] >= start_year]
    time_df = time_df[time_df['Movie_Release_Year'] <= end_year]
    if len(time_df) == 0:
        return []
    final_df = time_df[time_df['Movie_Revenue_Category'] == 'High']
    row_count = final_df.shape[0]
    if row_count < 10:
        final_df = time_df[(time_df['Movie_Revenue_Category'] == 'High') 
                            & (time_df['Movie_Revenue_Category'] == 'High_Med')]
        row_count = final_df.shape[0]
        if row_count < 10:
            final_df = time_df[(time_df['Movie_Revenue_Category'] == 'High') 
                            & (time_df['Movie_Revenue_Category'] == 'High_Med')
                            & (time_df['Movie_Revenue_Category'] == 'Low_Med')]
            row_count = final_df.shape[0]
            if row_count < 10:
                final_df = time_df
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
    
#print(RecommendTimePeriodBasedMovies(1993,2000))
