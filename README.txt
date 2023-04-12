DTI 5125 Final Project
######################

Explore-Exploit approach to Conversational Movie Recommender System
===================================================================

Dataset
-------
As part of the final project we are using CMU Movie Summary Corpus. This dataset contains information like:
1. Movie Box Office Revenue
2. Movie Run Time
3. Movie Genre
4. Movie Country
5. Movie Language
6. Movie Plot
7. Movie Name
etc.

Files Description
-----------------
1. ./MovieSummaries/CountryInfo.csv
Contains the information about all the countries which were included in the dataset and their respective occurances.

2. ./MovieSummaries/GenreInfo.csv
Contains the information about all the genres which were included in the dataset and their respective occurances.

3. ./MovieSummaries/LanguageInfo.csv
Contains the information about all the languages which were included in the dataset and their respective occurances.

4. ./MovieSummaries/MovieNames.csv
Contains the list of movie name which are included as part of the final dataset.

5. ./MovieSummaries/FinalDataset.csv
Contains the initial level of features which were present in the origianl dataset. Like: Movie Name, Movie Run Time, 
Movie Plot, Movie Genre (One Hot Encoding), Movie Language (One Hot Encoding), Movie Country (One Hot Encoding), etc.

6. ./MovieSummaries/MoviePlotFeatures.csv
Contains the features which were extracted from the movie plot. Here first Bag of Words was applied followed by 
TF-IDF and finally SVD for dimensionality reduction. The features which were generated after SVD are writen in this
file.

7. ./MovieSummaries/FinalFeatures.csv
Contains the combined features from both the above given file i.e. FinalDataset.csv and MoviePlotFeatures.csv.

8. ./PreProcessData.py
Contains code which cleans the dataset and keep only the relavent information which which be needed for classification,
clustering and finally recommendation. Here we have also done tasks like one hot encoding of the categorical
attributes. As output to this file we generate the ./MovieSummaries/MoviePlotFeatures.csv file.

9. ./FeatureEngineering.py
Contains code to process the movie plot and feature engineer. Here we use the movie plot to get Bag of Words on which
TF-IDF is applied and the for dimensionality reduction we use SVD and generate the ./MovieSummaries/MoviePlotFeatures.csv
file.

10. ./DatasetClustering.py
Here we use clustering to create a new feature i.e. using all the features we have created earlier we now apply clustering
on them to create a new feature which will be nothing but the cluster label. This file combines all the features into 
one file and write it into ./MovieSummaries/FinalFeatures.csv file.

11. ./DatasetVisualization.py
File contains the code for dataset visualization which can be used to infer more informations.

12. ./MovieRevenueClassification.py
This file contains the classification model and there comparision which will be used to classify a new movie to its 
corresponding Movie Revenue Category class which is an important aspect to our recommender system. This will help solve
the problem of cold start in recommender system.

13. ./GenreBasedRecommendation.py
The code to create recommendation based on the genre is present in this file.

14. ./KeywordBasedRecommendation.py
The code to create recommedation based on the keyword is present in this file.

15. ./LanguageBasedRecommendation.py
The code to create recommedation base on the language is present in this file.

16. ./MovieSimilarityBasedRecommendation.py
The code to create recommendation based on the movie similarity is present in this file.

17. ./TimePeriodBasedRecommendation.py
The code to create recommendation based on the given time period is present in this file.

18. ./RecommenderSystemServer.py
The code to host a server which will act as a back end to our recommender system is written in this file.

19. ./MovieSummaries/movie.metadata.tsv
This file is from CMU Movie Summary Corpus and contains the information about all the movies which are present in the
dataset like Movie Name, Movie ID, etc.

20. ./MovieSummaries/plot_summaries.txt
This file is from CMU Movie Summary Corpus and contains the information about all the movie plots which are present in
dataset.

Dependencies
------------
numpy
pandas
sister
scipy
scikit-learn
nltk
seaborn
matplotlib
wordcloud
re
json
parse
flask
ngrok

Steps for execution
-------------------
1. We start with pre processing the dataset so first run "python .\PreProcessData.py".
2. Next we need to do feature engineering from the movie plot which is given so we run "python .\FeatureEngineering.py" 
   and "python .\DatasetClustering.py" in the same order.
3. Now we can visualizate the dataset which is created so we run "python .\DatasetVisualization.py".
4. To deal with cold start we now need to predict the Revenue category class of the new movie so that we have relevant
   information on it which can be used in the recommendation system. For this we run "python .\MovieRevenueClassification.py".
5. Now we are all set to start our RecommenderSystemServer so we run "python .\RecommenderSystemServer.py". This will
   start the server on 2412 port number on localhost.
6. In order for the local server to be accessable to from the dialog flow now we use ngrok and open the PORT 2412
   for tunnelling using command "ngrok http 2412"
7. This should make everything available from the dialogflow to be accessed.
