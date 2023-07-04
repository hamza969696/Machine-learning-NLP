import numpy as np
import pandas as pd
import json
from sentence_transformers import SentenceTransformer # This requires an install
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeClassifier
from imblearn.over_sampling import RandomOverSampler

# Note: All codes using functions from sklearn were found on https://scikit-learn.org/stable/

def sentence_embedder(data, text, model): #Code inspired from https://www.sbert.net/
    """
    This function generates sentence embeddings for a column of text in a pandas dataframe.
    It needs three paramater the first paramater (data), should be a pandas dataframe.
    The second paramater should be a string specifying the column name of the column that contains the text to be transformed.
    The third parameter should be the model that will be used to transform the text into sentence embeddings.

    Returns the modified dataset and a dataset which contains the sentence embeddings seperatly
    """
    sentences = data[text].to_numpy().tolist()
    embeddings = model.encode(sentences)
    embeddingsdf = pd.DataFrame(embeddings, columns = [text + "_" + str(i) for i in range(len(embeddings[0]))])
    data = data.reset_index(drop=True)
    embeddingsdf = embeddingsdf.reset_index(drop=True)
    data = pd.concat([data, embeddingsdf], axis = 1)
    return data, embeddingsdf

def tfidf(data, name, model):
    """
    This function generates TFIDF features for a column of text in a pandas dataframe.
    It needs three paramater the first paramater (data), should be a pandas dataframe.
    The second paramater should be a string specifying the column name of the column that contains the text to be transformed.
    The third parameter should be the model that will be used to transform the text into TFIDF features.

    Returns the modified dataset and a dataset which contains TFIDF feature seperatly
    """
    corpus = data["abstract"].to_numpy()
    vector_abstract = model.transform(corpus)
    vector_abstract_df = pd.DataFrame.sparse.from_spmatrix(vector_abstract)
    col_names = [name + "_" + str(i) for i in range(len(vector_abstract_df.columns))]
    vector_abstract_df = vector_abstract_df.set_axis(col_names, axis = 1)
    data = data.join(vector_abstract_df)
    return data, vector_abstract_df

def topic_modelling(data, model, vectorizer): 
    """
    This function uses a unsupervised machine learning algorithm to create an amount of topics specified by the model from a given text,
    And give a value for observations containing how similar the observation is to each of the topics.
    It expects two paramters, the first parameter (data) should be a pandas dataframe that needs to be transformed.
    The second paramter (model) should be the model that is used to transform the data into topics.
    
    Returns the modified dataset and a dataset which contains the topic feature seperatly
    """
    corpus = data["abstract"]
    X = vectorizer.transform(corpus)
    transformed = model.transform(X)
    topicdf = pd.DataFrame(transformed, columns = ["topic" + str(i) for i in range(len(transformed[0]))])
    return data.join(topicdf), topicdf

def encoder(data, model):

    """
    This function one hot encodes the venue column from a pandas datset.
    It needs two parameters, a data paramter which contains pandas dataframe that needs to be transformed and the fitted OneHotEncoder model.

    It returns a modified pandas dataframe
    """
    ohe=model.transform(data.venue.values.reshape(-1,1)).toarray()
    dfOneHot=pd.DataFrame(ohe, columns = ["venue_"+str(model.categories_[0][i])
                                        for i in range(len(model.categories_[0]))])
    data = data.reset_index(drop=True)
    dfOneHot = dfOneHot.reset_index(drop=True)

    data=pd.concat([data, dfOneHot], axis=1)
    return data

#-------------------- Loading in the training data ----------------------------+
with open('train.json') as file:
    train=json.load(file)
    file.close()

train = pd.read_json('train.json')   

#-------------------- Modifying and splitting training data ----------------------------
    #Turn string to lower case and merge title and abstract in new column
train["title"] = train["title"].str.lower()
train["abstract"] =  train["abstract"].str.lower()
train["text"] = train["title"] + " " + train["abstract"] 


    #Split train data into training and validation data for both Y variables and the X feauture set.      
train['counts'] = train.groupby('authorId')['authorId'].transform('count')
selection  = train["counts"]>1 # Classes that appear once cannot be used in stratified sampling, will be added to the train set.
train = train.drop("counts", axis = 1, errors=['ignore'])
y = train.pop("authorId")
x_train, x_validation, y_train, y_validation = train_test_split(train[selection], y[selection], stratify=y[selection], test_size = 3418, random_state=1234) #Stratified sampling to ensure all classes are in the test set
x_train = pd.concat([x_train, train[~selection]])
y_train = pd.concat([y_train, y[~selection]])

#-------------------- Creating and fitting helper models ----------------------------
sentence_model = SentenceTransformer('all-mpnet-base-v2')  # Note: This will download the model on your system
counts=x_train.venue.value_counts()
train1=x_train.loc[x_train['venue'].isin(counts.index[counts>5]) & x_train['venue'].isin(counts.index[counts<1500])]
x_train['venue_encoded']=train1['venue']
x_train['venue_encoded']=  x_train['venue_encoded'].fillna('Other')
ohe_model = OneHotEncoder(handle_unknown="ignore").fit(x_train.venue_encoded.values.reshape(-1,1))
tfidf_model_word =  TfidfVectorizer(max_df = .9, max_features = 120 ).fit(x_train["text"].to_numpy()) 
tfidf_model_ngrams =  TfidfVectorizer(max_df = .9, min_df = 4 ,max_features= 100 , ngram_range = (2,3)).fit(x_train["text"].to_numpy()) 
vector_model = CountVectorizer().fit(x_train["text"].to_numpy())
topic_model = LatentDirichletAllocation(n_components=100).fit(vector_model.transform(x_train["text"].to_numpy())) 

#-------------------- Transforming datasets and feature engineering ----------------------------
x_train, sentences_tt = sentence_embedder(x_train, "title", sentence_model)
x_train, sentences_at = sentence_embedder(x_train, "abstract", sentence_model)
x_train, tfidf_word = tfidf(x_train, "word", tfidf_model_word)
x_train, tfidf_ngram = tfidf(x_train, "ngram", tfidf_model_ngrams)
x_train =  encoder(x_train, ohe_model)
x_train, train_topics = topic_modelling(x_train, topic_model, vector_model)

x_validation, sentences_tv = sentence_embedder(x_validation, "title", sentence_model)
x_validation, sentences_av = sentence_embedder(x_validation, "abstract", sentence_model)
x_validation, tfidf_word = tfidf(x_validation, "word", tfidf_model_word)
x_validation, tfidf_ngram = tfidf(x_validation, "ngram", tfidf_model_ngrams)
x_validation =  encoder(x_validation, ohe_model)
x_validation, train_topics = topic_modelling(x_validation, topic_model, vector_model)

x_train = x_train.drop(["paperId", "title", "authorName", "abstract", "venue", "text", "year", "venue_encoded"], axis = 1)
x_validation = x_validation.drop(["paperId", "title", "authorName", "abstract", "venue", "text", "year"], axis = 1)
ros = RandomOverSampler(random_state=0)
x_train, y_train_oversampled = ros.fit_resample(x_train, y_train) # Oversample to ensure balanced classes

for tr, te in zip(x_train.columns.tolist(), x_validation.columns.tolist()):
    if tr != te:
        print(tr, te) # Sanity check to see if all columns are the same in both train and validation data


pipeline = Pipeline([
    ('std_scalar', StandardScaler())
])

x_train_piped= pipeline.fit_transform(x_train)
x_validation_piped = pipeline.transform(x_validation)

#-------------------- Model training  ----------------------------
clf1 = RidgeClassifier(alpha=10)
clf1.fit(x_train_piped, y_train_oversampled)
clf1.score(x_train_piped, y_train_oversampled)
clf1.score(x_validation_piped, y_validation)

#-------------------- Generating predictions for the test data ----------------------------
    #Use the full dataset again
with open('train.json') as file:
    train=json.load(file)
    file.close()
x_train = pd.read_json('train.json')  
x_train["title"] = x_train["title"].str.lower()
x_train["abstract"] =  x_train["abstract"].str.lower()
x_train["text"] = x_train["title"] + " " + x_train["abstract"] 
y_train = x_train.pop("authorId") 

   #Load in the test set
with open('test.json') as file:
    test=json.load(file)
    file.close()
test = pd.read_json('test.json')
test["title"] = test["title"].str.lower()
test["abstract"] =  test["abstract"].str.lower()
test["text"] = test["title"] + " " + test["abstract"]
paperId = test.pop("paperId")

    #Reinitialize and fit the models on the full train set.
counts=x_train.venue.value_counts()
train1=x_train.loc[x_train['venue'].isin(counts.index[counts>5]) & x_train['venue'].isin(counts.index[counts<2000])]
x_train['venue_encoded']=train1['venue']
x_train['venue_encoded']=  x_train['venue_encoded'].fillna('Other')
ohe_model = OneHotEncoder(handle_unknown="ignore").fit(x_train.venue_encoded.values.reshape(-1,1)) 
tfidf_model_word =  TfidfVectorizer(max_df = .9, max_features = 120 ).fit(x_train["text"].to_numpy()) 
tfidf_model_ngrams =  TfidfVectorizer(max_df = .9, min_df = 4 ,max_features= 100 , ngram_range = (2,3)).fit(x_train["text"].to_numpy())
vector_model = CountVectorizer().fit(x_train["text"].to_numpy())
topic_model = LatentDirichletAllocation(n_components=100).fit(vector_model.transform(x_train["text"].to_numpy()))

    #Transform both the train and test dataset.
x_train, sentences_tt = sentence_embedder(x_train, "title", sentence_model)
x_train, sentences_at = sentence_embedder(x_train, "abstract", sentence_model)
x_train, tfidf_word = tfidf(x_train, "word", tfidf_model_word)
x_train, tfidf_ngram = tfidf(x_train, "ngram", tfidf_model_ngrams)
x_train =  encoder(x_train, ohe_model)
x_train, train_topics = topic_modelling(x_train, topic_model, vector_model)

test, sentences_tv = sentence_embedder(test, "title", sentence_model)
test, sentences_av = sentence_embedder(test, "abstract", sentence_model)
test, tfidf_word = tfidf(test, "word", tfidf_model_word)
test, tfidf_ngram = tfidf(test, "ngram", tfidf_model_ngrams)
test =  encoder(test, ohe_model)
test, train_topics = topic_modelling(test, topic_model, vector_model)

x_train = x_train.drop(["paperId", "title", "authorName", "abstract", "venue", "text", "year", "venue_encoded"], axis = 1)
test = test.drop(["title", "abstract", "venue", "text", "year"], axis = 1)

ros = RandomOverSampler(random_state=0)
x_train, y_train_oversampled = ros.fit_resample(x_train, y_train) # Oversample to ensure balanced classes

for tr, te in zip(x_train.columns.tolist(), test.columns.tolist()):
    if tr != te:
        print(tr, te) # Sanity check to see if all columns are the same in both train and test data


pipeline = Pipeline([
    ('std_scalar', StandardScaler())
])

x_train_piped= pipeline.fit_transform(x_train)
test_piped = pipeline.transform(test)

    #Model training
clf2 = RidgeClassifier(alpha=10)
clf2.fit(x_train_piped, y_train_oversampled)
    #Generating predictions and transforming it to submission file
predictions_1 = clf1.predict(test)
predictions_2 = clf2.predict(test)
result = []
for i, paper in enumerate(paperId):
    d = {"paperId" : paper, "authorId": str(predictions_2[i])} # Specify which predictions are to be transformed to a JSON file
    result.append(d)

with open('predicted.json', 'w') as outfile:
    json.dump(result, outfile)