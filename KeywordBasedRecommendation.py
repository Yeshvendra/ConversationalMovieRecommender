################################################################
# File Description:
# The code to create recommedation based on the keyword is present in this file.
################################################################

import sister
import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize as tokenize
from nltk.corpus import stopwords
from scipy.spatial.distance import cosine

#############SECTION FOR GLOBAL VARIABLES##############
CSV_FILE_NAME = './MovieSummaries/FinalFeatures.csv'
# Initialize the ALBERT word embedding module
EMBEDDER = sister.MeanEmbedding(lang="en")
LOG_CONFIDENCE = 1
lemmatizer = WordNetLemmatizer()
projectDataframe = pd.DataFrame()
#######################################################

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Function for clean the given text by removing stopwords and lemmatization
def cleanText(plot):
    
    # Tokenizing the book word by word
    plot_words = nltk.word_tokenize(plot)
    
    # Removal of punctuation
    partition_without_punct = [word.lower() for word in plot_words if word.isalpha()]

    # Stopwords Removal
    stop_words = set(stopwords.words('english'))
    partition_without_stopwords = [word for word in partition_without_punct if word not in stop_words]
    
    # Lemmatization
    partition_lemmat = [lemmatizer.lemmatize(word) for word in partition_without_stopwords]

    return ' '.join(partition_lemmat)

def ReadDataset():
    # Read csv file and create a dataframe
    dataFrame = pd.read_csv(CSV_FILE_NAME, usecols = ['Movie_Name','Movie_Plot'])
    return dataFrame

def PreProcessDataset():
    global projectDataframe  
    # Remove any row which has NA in it
    projectDataframe = projectDataframe.dropna()

    # Preprocess text from data for later use
    moviePlotEmbeddingList = []
    for plot in projectDataframe['Movie_Plot']:
        plotCleaned = cleanText(plot)
        moviePlotEmbeddingList.append(EMBEDDER(plotCleaned))
    
    projectDataframe['Movie_Plot_Embedding'] = moviePlotEmbeddingList

    return projectDataframe

def FindCosineSimilarity(embeddedInputQuery, processedDatasetEmbedding):
    cosineSimilarity = cosine(embeddedInputQuery, processedDatasetEmbedding)
    return cosineSimilarity

# Calculate similarity and accordingly return the response
def RecommendKeywordBasedMovies(inputQuery, top = 10):

    global projectDataframe
    preprocessedInputQuery = cleanText(inputQuery)

    embeddedInputQuery = EMBEDDER(preprocessedInputQuery)

    results = [FindCosineSimilarity(embeddedInputQuery, processedDatasetEmbedding) 
              for processedDatasetEmbedding in projectDataframe['Movie_Plot_Embedding']]
    np_results = np.array(results)
    movieList = []
        
    ind = np_results.argsort()[:top]
    for i in ind:
        row = projectDataframe.iloc[i]
        movieList.append(row['Movie_Name'])
    return movieList

projectDataframe = ReadDataset()
PreProcessDataset()
