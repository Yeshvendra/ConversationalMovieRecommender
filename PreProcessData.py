################################################################
# File Description:
# Contains code which cleans the dataset and keep only the relavent information which which be needed for classification,
# clustering and finally recommendation. Here we have also done tasks like one hot encoding of the categorical
# attributes. As output to this file we generate the ./MovieSummaries/MoviePlotFeatures.csv file.
################################################################

import pandas as pd
import numpy as np
import json
from dateutil.parser import parse
import re

df = pd.read_csv("./MovieSummaries/movie.metadata.tsv", sep ='\t', header= None)
df.columns = ['Movie_ID','Freebase_Movie_ID','Movie_Name','Movie_Release_Date','Movie_Box_Office_Revenue','Movie_Runtime','Movie_Languages','Movie_Countries','Movie_Genres']
df.dropna(inplace=True)
print(df.head())

row_count = df.shape[0]
print('Number of Records: ',row_count)

# Working on fetching the movie plot into dataframe
moviePlotList = []
moviePlotFileLineList = []
moviePlotFile = open('./MovieSummaries/plot_summaries.txt', "r", encoding="utf8")
for line in moviePlotFile:
    moviePlotFileLineList.append(line)
moviePlotFile.close()

for row in df['Movie_ID']:
    plotLine = ''
    for line in moviePlotFileLineList:
        rowStr = str(row)
        if rowStr in line:
            plotLine = line
            break
    if plotLine != '':
        plotLineList = re.split("\t", plotLine)
        plotLineList[1].replace("/n", "")
        plotLineList[1].replace("\\","")
        moviePlotList.append(plotLineList[1])
    else:
        moviePlotList.append('')

df["Movie_Plot"] = moviePlotList

# Removing row with empty Movie_Plot columns
df.drop(df[df["Movie_Plot"] == ''].index,inplace=True)
row_count = df.shape[0]
print('Number of Records: ',row_count)

# Write movie names in a csv file
movieNameList = []
for row in df['Movie_Name']:
    movieNameList.append(row)

movieNameDataframe = pd.DataFrame(movieNameList)
movieNameDataframe.to_csv('./MovieSummaries/MovieNames.csv')

# Bagging Movies based on Box Office Revenue

df['Movie_Revenue_Category'], cutbin = pd.qcut(df['Movie_Box_Office_Revenue'], 4, labels=['Low','Low_Med','High_Med','High'], retbins=True)
print(cutbin)

# Clean-Up Movie_Languages column
languageDictList = []
for row in df['Movie_Languages']:
    rowDict = json.loads(row)
    for key, value in rowDict.items():
        value = value.lower()
        rowDict[key] = value.replace("language","").strip()
    languageDictList.append(rowDict)

df['Movie_Languages_Dict'] = languageDictList

languageList = []
languageDict = {}
for row in df['Movie_Languages_Dict']:
    for eachLanguage in row.values():
        if eachLanguage in languageDict:
            languageDict[eachLanguage] += 1
        else:
            languageDict[eachLanguage] = 1
        languageList.append(eachLanguage)

languageSet = set(languageList)

print("Total Number of Languages: ", len(languageSet))
languageDataframe = pd.DataFrame(languageDict.items(), columns=['Language Name', 'Language Occurance'])
languageDataframe.to_csv('./MovieSummaries/LanguageInfo.csv')


for language in languageSet:
    languageColumn = []
    for row in df['Movie_Languages_Dict']:
        if language in row.values():
            languageColumn.append(1)
        else:
            languageColumn.append(0)
    df[language] = languageColumn

# Clean-Up Movie_Countries column
countryDictList = []
for row in df['Movie_Countries']:
    rowDict = json.loads(row)
    for key, value in rowDict.items():
        value = value.lower()
        rowDict[key] = value
    countryDictList.append(rowDict)

df['Movie_Countries_Dict'] = countryDictList

countryList = []
countryDict = {}
for row in df['Movie_Countries_Dict']:
    for eachCountry in row.values():
        if eachCountry in countryDict:
            countryDict[eachCountry] += 1
        else:
            countryDict[eachCountry] = 1
        countryList.append(eachCountry)

countrySet = set(countryList)

print("Total Number of Countries: ", len(countrySet))
countryDataframe = pd.DataFrame(countryDict.items(), columns=['Country Name', 'Country Occurance'])
countryDataframe.to_csv('./MovieSummaries/CountryInfo.csv')

for country in countrySet:
    countryColumn = []
    for row in df['Movie_Countries_Dict']:
        if country in row.values():
            countryColumn.append(1)
        else:
            countryColumn.append(0)
    df[country] = countryColumn

# Clean-Up Movie_Genres column
genreDictList = []
for row in df['Movie_Genres']:
    rowDict = json.loads(row)
    for key, value in rowDict.items():
        value = value.lower()
        rowDict[key] = value
    genreDictList.append(rowDict)

df['Movie_Genres_Dict'] = genreDictList

genreList = []
genreDict = {}
for row in df['Movie_Genres_Dict']:
    for eachGenre in row.values():
        if eachGenre in genreDict:
            genreDict[eachGenre] += 1
        else:
            genreDict[eachGenre] = 1
        genreList.append(eachGenre)

genreSet = set(genreList)

print("Total Number of Genres: ", len(genreSet))
genreDataframe = pd.DataFrame(genreDict.items(), columns=['Genre Name', 'Genre Occurance'])
genreDataframe.to_csv('./MovieSummaries/GenreInfo.csv')

for genre in genreSet:
    genreColumn = []
    for row in df['Movie_Genres_Dict']:
        if genre in row.values():
            genreColumn.append(1)
        else:
            genreColumn.append(0)
    df[genre] = genreColumn

# Working on Release Data Column
releaseYearList = []
for row in df['Movie_Release_Date']:
    year = parse(row, fuzzy=True).year
    releaseYearList.append(year)

df['Movie_Release_Year'] = releaseYearList

# Drop the columns which will no more be needed
df = df.drop(columns=['Freebase_Movie_ID','Movie_Release_Date','Movie_Box_Office_Revenue','Movie_Languages','Movie_Languages_Dict','Movie_Countries','Movie_Countries_Dict','Movie_Genres','Movie_Genres_Dict'])

# Write the DF into CSV file
df.to_csv('./MovieSummaries/FinalDataset.csv')

