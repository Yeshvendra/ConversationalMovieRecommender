################################################################
# File Description:
# Here we use clustering to create a new feature i.e. using all the features we have created earlier we now apply clustering
# on them to create a new feature which will be nothing but the cluster label. This file combines all the features into 
# one file and write it into ./MovieSummaries/FinalFeatures.csv file.
################################################################

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score,pairwise_distances
import seaborn as sns
sns.set_style('darkgrid')

finalDataset_df = pd.read_csv("./MovieSummaries/FinalDataset.csv")
finalDataset_df = finalDataset_df.drop(columns=['Unnamed: 0'])
finalDataset_df_copy = finalDataset_df.drop(columns=['Movie_ID', 'Movie_Plot', 'Movie_Revenue_Category', 'Movie_Name'])

moviePlot_df = pd.read_csv("./MovieSummaries/MoviePlotFeatures.csv")
moviePlot_df_copy = moviePlot_df.drop(columns=['Unnamed: 0'])

df = pd.concat([finalDataset_df_copy, moviePlot_df_copy], axis=1)

SSE = []
for k in range(1,15):
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(df)
    SSE.append(kmeans.inertia_)

plt.plot(range(1,15), SSE, 'bx-')
plt.title('Elbow Method')
plt.xlabel('Cluster Numbers')
plt.show()

kmeans = KMeans(n_clusters=6, init='k-means++', max_iter=300, n_init=10, random_state=0)
clusters = kmeans.fit_predict(df)
finalDataset_df["Cluster_Feature"] = clusters

clusterFeature_df = pd.concat([finalDataset_df, moviePlot_df_copy], axis=1)
#clusterFeature_df.to_csv('./MovieSummaries/FinalFeatures.csv')

# Calculate the silhouette score
silhouette_avg = silhouette_score(df, clusters)
print("The average silhouette score is :", silhouette_avg)

# Calculate the Calinski-Harabasz index
ch_score = calinski_harabasz_score(df, clusters)
print("The Calinski-Harabasz index is :", ch_score)

# Calculate the average distance between each point and its cluster centroid
closeness = []
for i in range(6):
    cluster_points = df[clusters == i]
    centroid = kmeans.cluster_centers_[i]
    dist = pairwise_distances(cluster_points, [centroid])
    closeness.append(dist.mean())

# Calculate the distance between each pair of cluster centroids
separation = pairwise_distances(kmeans.cluster_centers_)

# Print the results
print(f"Closeness within Clusters: {sum(closeness)/len(closeness)}")
print(f"Separation between Clusters: {separation.mean()}")


def visualization(df):
    # Create a new column in the dataframe to hold the cluster labels
    df['Cluster'] = clusters

    # Create a scatterplot using seaborn
    fig1=sns.scatterplot(data=df, x='Movie_Release_Year', y='Movie_Runtime', hue='Cluster', palette='bright')
    fig1.set(xlim=(1900, 2000))

    # Add axis labels and a title
    plt.xlabel('Movie_Release_Year')
    plt.ylabel('Movie_Runtime')
    plt.title('Cluster Plot')

    # Show the plot
    plt.show()

    fig2=sns.stripplot(data=df, x='Movie_Revenue_Category', y='Movie_Release_Year', hue='Cluster', palette='bright')
    fig2.set(ylim=(1900, 2000))

    # Add axis labels and a title
    plt.xlabel('Movie_Revenue_Category')
    plt.ylabel('Movie_Release_Year')
    plt.title('Cluster Plot 2')
    plt.show()

df=clusterFeature_df
#visualization(df)
