################################################################
# File Description:
# File contains the code for dataset visualization which can be used to infer more informations.
################################################################

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./MovieSummaries/FinalDataset.csv")
df = df[['Movie_Revenue_Category', 'Movie_Runtime', 'Movie_Release_Year', 'comedy', 'action', 'adventure', 'thriller', 'epic']]
# df = df[df['Movie_Release_Year'] != 1010]

# Visualizing Movies based on Year of release
fig1 = df.groupby('Movie_Release_Year').size().plot(kind='bar')
fig1.set(xlim=(1800, 2015))

plt.title("Histogram of movies based on Year of release")
plt.xlabel("Year of Release")
plt.ylabel("Number of Movies")
plt.show()

# Visualizing Movies based on 5 genre
fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
genres = ['comedy', 'action', 'adventure', 'thriller', 'epic']
values = [df[df['comedy'] == 1].shape[0],
          df[df['action'] == 1].shape[0],
          df[df['adventure'] == 1].shape[0],
          df[df['thriller'] == 1].shape[0],
          df[df['epic'] == 1].shape[0]]
plt.bar(genres, values, color ='maroon',
        width = 0.4)
 
plt.xlabel("Genres")
plt.ylabel("Number of Movies")
plt.title("Histogram of movies based on movie genres")
plt.show()

# Create covariance matrix for visualization
plt.matshow(df.corr())
plt.title("Covariance Matrix on some of the Attributes")
plt.show()
