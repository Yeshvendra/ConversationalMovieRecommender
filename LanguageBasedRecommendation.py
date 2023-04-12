################################################################
# File Description:
# The code to create recommedation base on the language is present in this file.
################################################################

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

df = pd.read_csv("./MovieSummaries/FinalFeatures.csv")
df = df.drop(columns=['Unnamed: 0'])

def RecommendLanguageBasedMovies(language, top = 10):
    language = language.lower()
    if language in df.columns:
        language_df = df[df[language] == 1]
        final_df = language_df[language_df['Movie_Revenue_Category'] == 'High']
        row_count = final_df.shape[0]
        if row_count < 10:
            final_df = language_df[(language_df['Movie_Revenue_Category'] == 'High') 
                                & (language_df['Movie_Revenue_Category'] == 'High_Med')]
            row_count = final_df.shape[0]
            if row_count < 10:
                final_df = language_df[(language_df['Movie_Revenue_Category'] == 'High') 
                                & (language_df['Movie_Revenue_Category'] == 'High_Med')
                                & (language_df['Movie_Revenue_Category'] == 'Low_Med')]
                row_count = final_df.shape[0]
                if row_count < 10:
                    final_df = language_df
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

#print(RecommendLanguageBasedMovies("marathi"))
