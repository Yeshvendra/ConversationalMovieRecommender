################################################################
# File Description:
# This file contains the classification model and there comparision which will be used to classify a new movie to its 
# corresponding Movie Revenue Category class which is an important aspect to our recommender system. This will help solve
# the problem of cold start in recommender system.
################################################################

import pandas as pd
import re 
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict, GridSearchCV
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE

ml_model = "GNB"  # The machine learning model can be chosen from among KNN, RF, and GNB

lemmatizer = WordNetLemmatizer()
label_encoder = LabelEncoder()

df = pd.read_csv("./MovieSummaries/FinalFeatures.csv")

regex = re.compile(r"\[|\]|<", re.IGNORECASE)
df.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in df.columns.values]

df['Movie_Revenue_Category'] = label_encoder.fit_transform(df['Movie_Revenue_Category'])

df = df.drop(columns=["Unnamed: 0", 'Movie_ID', 'Movie_Plot', 'Movie_Name'])

# Applying model with the cross validation technique
def apply_model_with_cross_validation(model, independent_variables, dependent_variable, folds):
    results = cross_val_predict(estimator=model, X=independent_variables, y=dependent_variable,cv=folds)
    conf_mat = confusion_matrix(dependent_variable, results)
    class_rep = classification_report(dependent_variable, results)
    return {"Classification Report": class_rep,"Confusion Matrix": conf_mat,"Prediction Results": results}


print("Applying Machine Learning model with 10 fold cross validation...")
model = ""
parameters = {}
if (ml_model == "KNN"):
    # Creating KNN classifier to be applied
    model = KNeighborsClassifier()
    parameters = {'n_neighbors': [1, 10],
                  'weights': ('uniform', 'distance')}
    print("Using k-Nearest Neighbor classifier...")
elif (ml_model == "GNB"):
    # Creating Gaussian Naive Bayes classifier to be applied
    model = GaussianNB()
    print("Using Gaussian Naive Bayes classifier...")
elif (ml_model == "RF"):
    # Creating Random Forest classifier to be applied
    model = RandomForestClassifier(random_state=13)
    print("Using Random Forest Classifier...")

# Setting up grid search for hyper parameter tunning
classifier = GridSearchCV(model, parameters)

result = apply_model_with_cross_validation(classifier, df, df['Movie_Revenue_Category'], 3)

print(result["Classification Report"])
print(result["Confusion Matrix"])

# Visualization of features using Scatter Plot
def visuaization():
    X_embedded = TSNE(n_components=2, learning_rate='auto',
                      init='random', perplexity=3).fit_transform(df)
    unique_revenue_category = df['Movie_Revenue_Category'].unique()
    unique_revenue_category_decoded = label_encoder.inverse_transform(unique_revenue_category)
    scat = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], marker="o", c=df['Movie_Revenue_Category'], s=40, edgecolor="k")
    classes = [l for l in unique_revenue_category_decoded]
    plt.legend(handles=scat.legend_elements()[0], labels=classes)
    plt.show()

visuaization()
