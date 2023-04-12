################################################################
# File Description:
# Contains code to process the movie plot and feature engineer. Here we use the movie plot to get Bag of Words on which
# TF-IDF is applied and the for dimensionality reduction we use SVD and generate the ./MovieSummaries/MoviePlotFeatures.csv
# file.
################################################################

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from wordcloud import WordCloud
import matplotlib.pyplot as plt

lemmatizer = WordNetLemmatizer()

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Function for clean the given text by removing stopwords and lemmatization
def cleanText(plot_words):
    # Removal of punctuation
    partition_without_punct = [word.lower() for word in plot_words if word.isalpha()]

    # Stopwords Removal
    stop_words = set(stopwords.words('english'))
    partition_without_stopwords = [word for word in partition_without_punct if word not in stop_words]
    
    # Lemmatization
    partition_lemmat = [lemmatizer.lemmatize(word) for word in partition_without_stopwords]
    return partition_lemmat

# Visualization of features using Word Cloud and Scatter Plot
def bow_visualization(bow_dataframe):
    print("Visualizing the features from BOW...")
    wc = WordCloud(background_color="white", width=1000, height=1000, max_words=100, relative_scaling=0.5,
                       normalize_plurals=False).generate_from_frequencies(bow_dataframe.sum())
    plt.imshow(wc)
    plt.show()
        

# Function for feature engineering
def featureEngineer(df):
    moviePlotCleanedList = []
    for plot in df['Movie_Plot']:
        # Tokenizing the book word by word
        plot_words = nltk.word_tokenize(plot)
        moviePlotCleanedList.append(' '.join(cleanText(plot_words)))
    
    df['Movie_Plot_Cleaned'] = moviePlotCleanedList

    print(df.head())

    print("Applying Bag of Words for Feature Engineering...")
    # Applying Bag Of Words for feature extraction with unigrams and bigrams
    vectorizer = CountVectorizer(ngram_range=(1,1))
    bow_transform = vectorizer.fit_transform(df['Movie_Plot_Cleaned'])
    bow_dataframe = pd.DataFrame(bow_transform.toarray(), columns=vectorizer.get_feature_names_out())
    print("Applying Bag of Words complete!!")

    print(bow_dataframe)

    print("Applying TF-IDF for Feature Engineering...")
    # Applying TFIDF for feature extraction
    transformer = TfidfTransformer()
    tfidf_transform = transformer.fit_transform(bow_dataframe)
    tfidf_dataframe = pd.DataFrame(tfidf_transform.toarray(), columns=vectorizer.get_feature_names_out())
    print("Applying TF-IDF complete!!")

    #print(tfidf_dataframe)

    # Visualizing the features
    bow_visualization(bow_dataframe)

    print("Applying Singular Value Decomposition for LSA...")
    # Applying Truncated SVD for LSA
    svd = TruncatedSVD(n_components=400, n_iter=2, random_state=42)
    svd_transform_array = svd.fit_transform(tfidf_dataframe)
    svd_dataframe = pd.DataFrame(svd_transform_array)
    print("Applying SVD complete!!")

    #df = pd.concat([df, svd_dataframe], axis=1)
    return svd_dataframe

df = pd.read_csv("./MovieSummaries/FinalDataset.csv")
df = df.drop(columns=["Unnamed: 0"])
print(df.head())
row_count = df.shape[0]
print("Total Number of Records are: ", row_count)

result_df = featureEngineer(df)
# Write the DF into CSV file
result_df.to_csv('./MovieSummaries/MoviePlotFeatures.csv')
print(result_df.head())
