
import pandas as pd
import numpy as np
import spacy
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import fbeta_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer

"""# Tokenization"""

# import spacy

# Load the medium english model.
# We will use this model to get embedding features for tokens later.
# !python -m spacy download en_core_web_md

nlp = spacy.load('en_core_web_sm', disable=['ner'])
nlp.remove_pipe('tagger')
nlp.remove_pipe('parser')

# Download a stopword list
import nltk

nltk.download('stopwords')


# @Tokenize
def spacy_tokenize(string):
    tokens = list()
    doc = nlp(string)
    for token in doc:
        tokens.append(token)
    return tokens


# @Normalize
def normalize(tokens):
    normalized_tokens = list()
    for token in tokens:
        normalized = token.text.lower().strip()
        if ((token.is_alpha or token.is_digit)):
            normalized_tokens.append(normalized)
    return normalized_tokens
    return normalized_tokens


# @Tokenize and normalize
def tokenize_normalize(string):
    return normalize(spacy_tokenize(string))


def evaluation_summary(description, predictions, true_labels):
    print("Evaluation for: " + description)
    precision = precision_score(predictions, true_labels, average='macro')
    recall = recall_score(predictions, true_labels, average='macro')
    accuracy = accuracy_score(predictions, true_labels)
    f1 = fbeta_score(predictions, true_labels, 1, average='macro')  # 1 means f_1 measure
    print("Classifier '%s' has Acc=%0.3f P=%0.3f R=%0.3f F1=%0.3f" % (description, accuracy, precision, recall, f1))
    # Specify three digits instead of the default two.
    print(classification_report(predictions, true_labels, digits=3))
    print('\nConfusion matrix:\n',
          confusion_matrix(true_labels, predictions))  # Note the order here is true, predicted, odd.


class DataFrameToArrayTransformer(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, X):
        # print(X.shape)
        # print(np.transpose(np.matrix(X)).shape)
        return np.transpose(np.matrix(X))


class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.    """

    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


"""## Defining and initializing classifiers."""

one_hot_vectorizer = CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)
dummy_clf = DummyClassifier(strategy="uniform")
dummy_clf1 = DummyClassifier(strategy="stratified")
# decisionTree= DecisionTreeClassifier(random_state=10)

dtaframe = pd.read_csv('finalTestData.csv', encoding='latin1')
firstcopy = secondcopy = dtaframe

firstcopy.head(2)

y = (firstcopy['label'])

len(y)

firstcopy.drop(columns='Unnamed: 0', inplace=True)
firstcopy.drop(columns='label', inplace=True)

# nan values are replaced by this
firstcopy["source"].fillna("Twitter Web App", inplace=True)
firstcopy["userLocation"].fillna("No Location", inplace=True)
firstcopy["URL"].fillna("https://twitter.com/home", inplace=True)

firstcopy.dropna(inplace=True)
y.dropna(inplace=True)

print(len(firstcopy), len(y))
dataset1 = []
counter = 0
for index, row in firstcopy.iterrows():
    new_row = ""
    new_row = new_row + str(row['userName']) + " " + str(row['text']) + " " + str(row['textLen']) + " " + str(
        row['retweetsCount']) + " " + str(row['favoriteCount']) + " " + str(row['source']) + " " + str(
        row['language']) + " " + str(row['favourited']) + " " + str(row['retweeted']) + " " + str(
        row['userLocation']) + " " + str(row['URL']) + " " + str(row['userfollowers_count']) + " " + str(
        row['userfriends_count']) + " " + str(row['userListed_count']) + " " + str(
        row['userFavorites_count']) + " " + str(row['userStatuses_count']) + " " + str(row['userVerified']) + " " + str(
        row['userProtected']) + " " + str(row['sentiment'])
    dataset1.append(new_row)
    # print(type(new_row))
    counter = counter + 1
    # Val_labels.append(new_label)

X_train, X_test, y_train, y_test = train_test_split(
    dataset1, y, test_size=0.3, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=0)

onehot_training_features = one_hot_vectorizer.fit_transform(X_train)
onehot_validation_features = one_hot_vectorizer.transform(X_val)
onehot_testing_features = one_hot_vectorizer.transform(X_test)

print(len(X_train))
print(len(X_val))
print(len(X_test))

"""# Models"""

dummy_stratified = Pipeline([
    ('countvectorizer', one_hot_vectorizer),
    ('dummy_stratified', dummy_clf1)])
dummy_stratified.fit(X_train, y_train)

evaluation_summary("Dummy Majority", dummy_stratified.predict(X_val), y_val)

svc_clf = SVC(gamma='auto')
svc__clf = Pipeline([
    ('one-hot', one_hot_vectorizer),
    ('SVC onehot', svc_clf)
])

type(X_train[0])

svc__clf.fit(X_train, y_train)

print(
    "\n*********************************************************TRAINING***********************************************************************************\n")
svc_train = evaluation_summary("SVC - TRAIN", svc__clf.predict(X_train), y_train)

print(
    "\n***********************************************************TESTING*********************************************************************************\n")
svc_test = evaluation_summary("SVC - TEST", svc__clf.predict(X_val), y_val)
print(
    "\n********************************************************************************************************************************************\n")

"""# **Data Analysis**"""

X_train1, X_test1, y_train1, y_test1 = train_test_split(
    firstcopy, y, test_size=0.3, random_state=0)
X_train1, X_val1, y_train1, y_val1 = train_test_split(
    X_train1, y_train1, test_size=0.25, random_state=0)

# onehot_training_features = one_hot_vectorizer.fit_transform(X_train1)
# onehot_validation_features = one_hot_vectorizer.transform(X_val1)
# onehot_testing_features = one_hot_vectorizer.transform(X_test1)

X_train1.head(2)

"""# SVC"""

pipeline_feature_00 = Pipeline([
    ('union', FeatureUnion(
        [
            ('text', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ]))
        ])

     ),
    ('svc', SVC(gamma='auto'))
])

pipeline_feature_00.fit(X_train1, y_train1)
result00 = pipeline_feature_00.predict(X_val1)
evaluation_summary("Results with one feature on validation set", result00, y_val1)

pipeline_feature_0 = Pipeline([
    ('union', FeatureUnion(
        [
            ('text', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),

            ('source', Pipeline([
                ('selector', ItemSelector(key='source')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ]))

        ])

     ),
    ('svc', SVC(gamma='auto'))
])

pipeline_feature_0.fit(X_train1, y_train1)
result0 = pipeline_feature_0.predict(X_val1)
evaluation_summary("Results with two feature on validation set\n \t\t", result0, y_val1)

pipeline_feature_1 = Pipeline([
    ('union', FeatureUnion(
        [
            ('text', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),

            ('textlen', Pipeline([
                ('selector', ItemSelector(key='textLen')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),

            ('source', Pipeline([
                ('selector', ItemSelector(key='source')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ]))

        ])

     ),
    ('svc', SVC(gamma='auto'))
])

pipeline_feature_1.fit(X_train1, y_train1)
result1 = pipeline_feature_1.predict(X_val1)
evaluation_summary("Results with two feature on validation set", result1, y_val1)

pipeline_feature_2 = Pipeline([
    ('union', FeatureUnion(
        [
            ('text', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),

            ('textlen', Pipeline([
                ('selector', ItemSelector(key='textLen')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),

            ('favoriteCount', Pipeline([
                ('selector', ItemSelector(key='favoriteCount')),
                ('Retweetarray', DataFrameToArrayTransformer()),
            ])),
            ('source', Pipeline([
                ('selector', ItemSelector(key='source')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ]))
        ])

     ),
    ('svc', SVC(gamma='auto'))
])

pipeline_feature_2.fit(X_train1, y_train1)
result2 = pipeline_feature_2.predict(X_val1)
evaluation_summary("Results with two feature on validation set", result2, y_val1)

"""**We can see how retweetsCount and language is decreasing the performance of the classifier**"""

pipeline_feature_3 = Pipeline([
    ('union', FeatureUnion(
        [
            ('text', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),

            ('textlen', Pipeline([
                ('selector', ItemSelector(key='textLen')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),
            ('retweetsCount', Pipeline([
                ('selector', ItemSelector(key='retweetsCount')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),

            ('language', Pipeline([
                ('selector', ItemSelector(key='language')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),
            ('favoriteCount', Pipeline([
                ('selector', ItemSelector(key='favoriteCount')),
                ('Retweetarray', DataFrameToArrayTransformer()),
            ])),
            ('source', Pipeline([
                ('selector', ItemSelector(key='source')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ]))
        ])

     ),
    ('svc', SVC(gamma='auto'))
])

pipeline_feature_3.fit(X_train1, y_train1)
result3 = pipeline_feature_3.predict(X_val1)
evaluation_summary("Results with two feature on validation set", result3, y_val1)

"""**By adding userfollowers we see some improvement when compared to the retweetedcount**"""

pipeline_feature_4 = Pipeline([
    ('union', FeatureUnion(
        [
            ('text', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),

            ('textlen', Pipeline([
                ('selector', ItemSelector(key='textLen')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),
            ('userfollowers_count', Pipeline([
                ('selector', ItemSelector(key='userfollowers_count')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),
            ('favoriteCount', Pipeline([
                ('selector', ItemSelector(key='favoriteCount')),
                ('Retweetarray', DataFrameToArrayTransformer()),
            ])),
            ('source', Pipeline([
                ('selector', ItemSelector(key='source')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ]))
        ])

     ),
    ('svc', SVC(gamma='auto'))
])

pipeline_feature_4.fit(X_train1, y_train1)
result4 = pipeline_feature_4.predict(X_val1)
evaluation_summary("Results with two feature on validation set", result4, y_val1)

"""**We see how adding 'userfriends_count','userListed_count','userFavorites_count','userStatuses_count'there is a great performance lost**"""

pipeline_feature_5 = Pipeline([
    ('union', FeatureUnion(
        [
            ('text', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),

            ('textlen', Pipeline([
                ('selector', ItemSelector(key='textLen')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),

            ('userStatuses_count', Pipeline([
                ('selector', ItemSelector(key='userStatuses_count')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),
            ('userFavorites_count', Pipeline([
                ('selector', ItemSelector(key='userFavorites_count')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),
            ('userListed_count', Pipeline([
                ('selector', ItemSelector(key='userListed_count')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),
            ('userfriends_count', Pipeline([
                ('selector', ItemSelector(key='userfriends_count')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),
            ('userfollowers_count', Pipeline([
                ('selector', ItemSelector(key='userfollowers_count')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),
            ('favoriteCount', Pipeline([
                ('selector', ItemSelector(key='favoriteCount')),
                ('Retweetarray', DataFrameToArrayTransformer()),
            ])),
            ('source', Pipeline([
                ('selector', ItemSelector(key='source')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ]))
        ])

     ),
    ('svc', SVC(gamma='auto'))
])

pipeline_feature_5.fit(X_train1, y_train1)
result5 = pipeline_feature_5.predict(X_val1)
evaluation_summary("Results with two feature on validation set", result5, y_val1)

pipeline_feature_6 = Pipeline([
    ('union', FeatureUnion(
        [
            ('text', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),

            ('textlen', Pipeline([
                ('selector', ItemSelector(key='textLen')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),
            ('URL', Pipeline([
                ('selector', ItemSelector(key='URL')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),
            ('userfollowers_count', Pipeline([
                ('selector', ItemSelector(key='userfollowers_count')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),
            ('favoriteCount', Pipeline([
                ('selector', ItemSelector(key='favoriteCount')),
                ('Retweetarray', DataFrameToArrayTransformer()),
            ])),
            ('source', Pipeline([
                ('selector', ItemSelector(key='source')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ]))
        ])

     ),
    ('svc', SVC(gamma='auto'))
])

pipeline_feature_6.fit(X_train1, y_train1)
result6 = pipeline_feature_6.predict(X_val1)
evaluation_summary("Results with two feature on validation set", result6, y_val1)

pipeline_feature_7 = Pipeline([
    ('union', FeatureUnion(
        [
            ('text', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),

            ('textlen', Pipeline([
                ('selector', ItemSelector(key='textLen')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),

            ('userLocation', Pipeline([
                ('selector', ItemSelector(key='userLocation')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),

            ('URL', Pipeline([
                ('selector', ItemSelector(key='URL')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),
            ('userfollowers_count', Pipeline([
                ('selector', ItemSelector(key='userfollowers_count')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),
            ('favoriteCount', Pipeline([
                ('selector', ItemSelector(key='favoriteCount')),
                ('array', DataFrameToArrayTransformer()),
            ])),
            ('sentiment', Pipeline([
                ('selector', ItemSelector(key='sentiment')),
                ('array', DataFrameToArrayTransformer()),
            ])),
            ('source', Pipeline([
                ('selector', ItemSelector(key='source')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ]))
        ])

     ),
    ('svc', SVC(gamma='auto'))
])

pipeline_feature_7.fit(X_train1, y_train1)

result7 = pipeline_feature_7.predict(X_val1)

evaluation_summary("Results with two feature on validation set", result7, y_val1)

"""# Decision Tree"""

pipeline_feature_01 = Pipeline([
    ('union', FeatureUnion(
        [
            ('text', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),

            ('textlen', Pipeline([
                ('selector', ItemSelector(key='textLen')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),

            ('userLocation', Pipeline([
                ('selector', ItemSelector(key='userLocation')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ]))
        ])

     ),
    ('dTree', DecisionTreeClassifier(random_state=10))
])

pipeline_feature_01.fit(X_train1, y_train1)
result01 = pipeline_feature_01.predict(X_val1)
evaluation_summary("Results with two feature on validation set", result01, y_val1)

pipeline_feature_02 = Pipeline([
    ('union', FeatureUnion(
        [
            ('text', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),

            ('URL', Pipeline([
                ('selector', ItemSelector(key='URL')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),
            ('userfollowers_count', Pipeline([
                ('selector', ItemSelector(key='userfollowers_count')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),
            ('favoriteCount', Pipeline([
                ('selector', ItemSelector(key='favoriteCount')),
                ('array', DataFrameToArrayTransformer()),
            ])),
            ('sentiment', Pipeline([
                ('selector', ItemSelector(key='sentiment')),
                ('array', DataFrameToArrayTransformer()),
            ])),
            ('source', Pipeline([
                ('selector', ItemSelector(key='source')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),

            ('textlen', Pipeline([
                ('selector', ItemSelector(key='textLen')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),

            ('userLocation', Pipeline([
                ('selector', ItemSelector(key='userLocation')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ]))
        ])

     ),
    ('dTree', DecisionTreeClassifier(random_state=10))
])

pipeline_feature_02.fit(X_train1, y_train1)
result02 = pipeline_feature_02.predict(X_val1)
evaluation_summary("Results with two feature on validation set", result02, y_val1)

"""# Random Forest"""

pipeline_feature_31 = Pipeline([
    ('union', FeatureUnion(
        [
            ('text', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),

            ('textlen', Pipeline([
                ('selector', ItemSelector(key='textLen')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),

            ('userLocation', Pipeline([
                ('selector', ItemSelector(key='userLocation')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ]))
        ])

     ),
    ('rTree', RandomForestClassifier(random_state=10))
])

pipeline_feature_31.fit(X_train1, y_train1)
result31 = pipeline_feature_31.predict(X_val1)
evaluation_summary("Results with two feature on validation set", result31, y_val1)

pipeline_feature_32 = Pipeline([
    ('union', FeatureUnion(
        [
            ('text', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),

            ('URL', Pipeline([
                ('selector', ItemSelector(key='URL')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),
            ('userfollowers_count', Pipeline([
                ('selector', ItemSelector(key='userfollowers_count')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),
            ('favoriteCount', Pipeline([
                ('selector', ItemSelector(key='favoriteCount')),
                ('array', DataFrameToArrayTransformer()),
            ])),
            ('sentiment', Pipeline([
                ('selector', ItemSelector(key='sentiment')),
                ('array', DataFrameToArrayTransformer()),
            ])),
            ('source', Pipeline([
                ('selector', ItemSelector(key='source')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),

            ('textlen', Pipeline([
                ('selector', ItemSelector(key='textLen')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),

            ('userLocation', Pipeline([
                ('selector', ItemSelector(key='userLocation')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ]))
        ])

     ),
    ('rTree', RandomForestClassifier(random_state=10))
])

pipeline_feature_32.fit(X_train1, y_train1)
result32 = pipeline_feature_32.predict(X_val1)
evaluation_summary("Results with two feature on validation set", result32, y_val1)

"""# BernoulliNB"""

pipeline_feature_41 = Pipeline([
    ('union', FeatureUnion(
        [
            ('text', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),

            ('textlen', Pipeline([
                ('selector', ItemSelector(key='textLen')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),

            ('userLocation', Pipeline([
                ('selector', ItemSelector(key='userLocation')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ]))
        ])

     ),
    ('Gaussian', BernoulliNB())
])

pipeline_feature_41.fit(X_train1, y_train1)

result41 = pipeline_feature_41.predict(X_val1)
evaluation_summary("Results with two feature on validation set", result41, y_val1)

pipeline_feature_42 = Pipeline([
    ('union', FeatureUnion(
        [
            ('text', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),

            ('URL', Pipeline([
                ('selector', ItemSelector(key='URL')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),
            ('userfollowers_count', Pipeline([
                ('selector', ItemSelector(key='userfollowers_count')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),
            ('favoriteCount', Pipeline([
                ('selector', ItemSelector(key='favoriteCount')),
                ('array', DataFrameToArrayTransformer()),
            ])),
            ('sentiment', Pipeline([
                ('selector', ItemSelector(key='sentiment')),
                ('array', DataFrameToArrayTransformer()),
            ])),
            ('source', Pipeline([
                ('selector', ItemSelector(key='source')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),

            ('textlen', Pipeline([
                ('selector', ItemSelector(key='textLen')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),

            ('userLocation', Pipeline([
                ('selector', ItemSelector(key='userLocation')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ]))
        ])

     ),
    ('Gaussian', BernoulliNB())
])

pipeline_feature_42.fit(X_train1, y_train1)
result42 = pipeline_feature_42.predict(X_val1)
evaluation_summary("Results with two feature on validation set", result42, y_val1)

"""# Adding new feature 1

VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text by C.J. Hutto and Eric Gilbert Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

# firstcopy
# sentences =
list_ = []

# Example code:
for sentence in firstcopy['text']:
    vs = analyzer.polarity_scores(sentence)
    list_.append(vs)
    # print("{:-<65} {}".format(sentence, str(vs)))

neg = pos = neu = compound = []
for i in list_:
    neg.append(i['neg'])
    pos.append(i['pos'])
    neu.append(i['neu'])
    compound.append(i['compound'])

data_frame = firstcopy

data_frame.head(2)

data_frame['negative'] = pd.DataFrame(neg, columns=['negative'])
data_frame['positive'] = pd.DataFrame(pos, columns=['positive'])
data_frame['compound'] = pd.DataFrame(compound, columns=['compound'])
data_frame['neutral'] = pd.DataFrame(neu, columns=['neutral'])

data_frame.head(5)

"""# Testing claasifier with new data"""

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    data_frame, y, test_size=0.3, random_state=0)
X_train2, X_val2, y_train2, y_val2 = train_test_split(
    X_train2, y_train2, test_size=0.25, random_state=0)

"""#Summary"""

pipeline_feature_51 = Pipeline([
    ('union', FeatureUnion(
        [
            ('text', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),
            ('URL', Pipeline([
                ('selector', ItemSelector(key='URL')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),
            ('userfollowers_count', Pipeline([
                ('selector', ItemSelector(key='userfollowers_count')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),
            ('favoriteCount', Pipeline([
                ('selector', ItemSelector(key='favoriteCount')),
                ('array', DataFrameToArrayTransformer()),
            ])),
            ('sentiment', Pipeline([
                ('selector', ItemSelector(key='sentiment')),
                ('array', DataFrameToArrayTransformer()),
            ])),
            ('source', Pipeline([
                ('selector', ItemSelector(key='source')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),
            ('neg', Pipeline([
                ('selector', ItemSelector(key='negative')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),
            ('pos', Pipeline([
                ('selector', ItemSelector(key='positive')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),
            ('neu', Pipeline([
                ('selector', ItemSelector(key='neutral')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),
            ('compound', Pipeline([
                ('selector', ItemSelector(key='compound')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),
            ('textlen', Pipeline([
                ('selector', ItemSelector(key='textLen')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),

            ('userLocation', Pipeline([
                ('selector', ItemSelector(key='userLocation')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ]))
        ])

     ),
    ('dummy_stratified', dummy_clf1)
])
pipeline_feature_52 = Pipeline([
    ('union', FeatureUnion(
        [
            ('text', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),
            ('URL', Pipeline([
                ('selector', ItemSelector(key='URL')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),
            ('userfollowers_count', Pipeline([
                ('selector', ItemSelector(key='userfollowers_count')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),
            ('favoriteCount', Pipeline([
                ('selector', ItemSelector(key='favoriteCount')),
                ('array', DataFrameToArrayTransformer()),
            ])),
            ('sentiment', Pipeline([
                ('selector', ItemSelector(key='sentiment')),
                ('array', DataFrameToArrayTransformer()),
            ])),
            ('source', Pipeline([
                ('selector', ItemSelector(key='source')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),
            ('neg', Pipeline([
                ('selector', ItemSelector(key='negative')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),
            ('pos', Pipeline([
                ('selector', ItemSelector(key='positive')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),
            ('neu', Pipeline([
                ('selector', ItemSelector(key='neutral')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),
            ('compound', Pipeline([
                ('selector', ItemSelector(key='compound')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),
            ('textlen', Pipeline([
                ('selector', ItemSelector(key='textLen')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),

            ('userLocation', Pipeline([
                ('selector', ItemSelector(key='userLocation')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ]))
        ])

     ),
    ('svc', SVC(gamma='auto'))
])

pipeline_feature_53 = Pipeline([
    ('union', FeatureUnion(
        [
            ('text', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),
            ('URL', Pipeline([
                ('selector', ItemSelector(key='URL')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),
            ('userfollowers_count', Pipeline([
                ('selector', ItemSelector(key='userfollowers_count')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),
            ('favoriteCount', Pipeline([
                ('selector', ItemSelector(key='favoriteCount')),
                ('array', DataFrameToArrayTransformer()),
            ])),
            ('sentiment', Pipeline([
                ('selector', ItemSelector(key='sentiment')),
                ('array', DataFrameToArrayTransformer()),
            ])),
            ('source', Pipeline([
                ('selector', ItemSelector(key='source')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),
            ('neg', Pipeline([
                ('selector', ItemSelector(key='negative')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),
            ('pos', Pipeline([
                ('selector', ItemSelector(key='positive')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),
            ('neu', Pipeline([
                ('selector', ItemSelector(key='neutral')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),
            ('compound', Pipeline([
                ('selector', ItemSelector(key='compound')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),
            ('textlen', Pipeline([
                ('selector', ItemSelector(key='textLen')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),

            ('userLocation', Pipeline([
                ('selector', ItemSelector(key='userLocation')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ]))
        ])

     ),
    ('dTree', DecisionTreeClassifier(random_state=10))
])

pipeline_feature_54 = Pipeline([
    ('union', FeatureUnion(
        [
            ('text', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),
            ('URL', Pipeline([
                ('selector', ItemSelector(key='URL')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),
            ('userfollowers_count', Pipeline([
                ('selector', ItemSelector(key='userfollowers_count')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),
            ('favoriteCount', Pipeline([
                ('selector', ItemSelector(key='favoriteCount')),
                ('array', DataFrameToArrayTransformer()),
            ])),
            ('sentiment', Pipeline([
                ('selector', ItemSelector(key='sentiment')),
                ('array', DataFrameToArrayTransformer()),
            ])),
            ('source', Pipeline([
                ('selector', ItemSelector(key='source')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),
            ('neg', Pipeline([
                ('selector', ItemSelector(key='negative')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),
            ('pos', Pipeline([
                ('selector', ItemSelector(key='positive')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),
            ('neu', Pipeline([
                ('selector', ItemSelector(key='neutral')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),
            ('compound', Pipeline([
                ('selector', ItemSelector(key='compound')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),
            ('textlen', Pipeline([
                ('selector', ItemSelector(key='textLen')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),

            ('userLocation', Pipeline([
                ('selector', ItemSelector(key='userLocation')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ]))
        ])

     ),
    ('rTree', RandomForestClassifier(random_state=10))
])

pipeline_feature_55 = Pipeline([
    ('union', FeatureUnion(
        [
            ('text', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),
            ('URL', Pipeline([
                ('selector', ItemSelector(key='URL')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),
            ('userfollowers_count', Pipeline([
                ('selector', ItemSelector(key='userfollowers_count')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),
            ('favoriteCount', Pipeline([
                ('selector', ItemSelector(key='favoriteCount')),
                ('array', DataFrameToArrayTransformer()),
            ])),
            ('sentiment', Pipeline([
                ('selector', ItemSelector(key='sentiment')),
                ('array', DataFrameToArrayTransformer()),
            ])),
            ('source', Pipeline([
                ('selector', ItemSelector(key='source')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),
            ('neg', Pipeline([
                ('selector', ItemSelector(key='negative')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),
            ('pos', Pipeline([
                ('selector', ItemSelector(key='positive')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),
            ('neu', Pipeline([
                ('selector', ItemSelector(key='neutral')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),
            ('compound', Pipeline([
                ('selector', ItemSelector(key='compound')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),
            ('textlen', Pipeline([
                ('selector', ItemSelector(key='textLen')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),

            ('userLocation', Pipeline([
                ('selector', ItemSelector(key='userLocation')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ]))
        ])

     ),
    ('Gaussian', BernoulliNB())
])

pipeline_feature_51.fit(X_train2, y_train2)
result51 = pipeline_feature_51.predict(X_val2)
evaluation_summary("Results with two feature on validation set", result51, y_val2)

pipeline_feature_52.fit(X_train2, y_train2)
result52 = pipeline_feature_52.predict(X_val2)
evaluation_summary("Results with two feature on validation set", result52, y_val2)

pipeline_feature_53.fit(X_train2, y_train2)
result53 = pipeline_feature_53.predict(X_val2)
evaluation_summary("Results with two feature on validation set", result53, y_val2)

pipeline_feature_54.fit(X_train2, y_train2)
result54 = pipeline_feature_54.predict(X_val2)
evaluation_summary("Results with two feature on validation set", result54, y_val2)

pipeline_feature_55.fit(X_train2, y_train2)
result55 = pipeline_feature_55.predict(X_val2)
evaluation_summary("Results with two feature on validation set", result55, y_val2)

"""## new features ideas
https://www.pluralsight.com/guides/building-features-from-text-data
"""

# df1= data_frame
data_frame['spl'] = data_frame['text'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import warnings

# %matplotlib inline
warnings.filterwarnings("ignore", category=DeprecationWarning)
from nltk.corpus import stopwords

stop = stopwords.words('english')

data_frame['processedtext'] = data_frame['text'].str.replace('[^\w\s]', '')
data_frame['processedtext'] = data_frame['processedtext'].apply(
    lambda x: " ".join(x for x in x.split() if x not in stop))
data_frame['processedtext'] = data_frame['processedtext'].apply(lambda x: " ".join(x.lower() for x in x.split()))

# Lines 4 to 6
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
data_frame['processedtext'] = data_frame['processedtext'].apply(
    lambda x: " ".join([stemmer.stem(word) for word in x.split()]))

"""# Adding new feature 2 
Entity
"""


stop = stopwords.words('english')
nlp = spacy.load("en_core_web_sm")

df = []
entities = []
numOfEntities = []
for i in data_frame['processedtext']:
    df.append(nlp(i))
for i in df:
    sent = ''
    counter = 0
    for word in i.ents:
        counter = counter + 1
        sent = sent + " " + word.label_
        # print(word.text,word.label_)
    entities.append(sent)
    numOfEntities.append(counter)

print(entities[0:5])
# numOfEntities[0:5]


data_frame['entities'] = pd.DataFrame(entities, columns=['entities'])
data_frame['numOfEntities'] = pd.DataFrame(numOfEntities, columns=['numOfEntities'])

data_frame.replace(r'^\s*$', "none", regex=True)

X_train22, X_test22, y_train22, y_test22 = train_test_split(
    data_frame, y, test_size=0.3, random_state=0)
X_train22, X_val22, y_train22, y_val22 = train_test_split(
    X_train22, y_train22, test_size=0.25, random_state=0)

"""https://stackabuse.com/python-for-nlp-sentiment-analysis-with-scikit-learn/

# Summary after adding spl and postcleantext
"""

pipeline_feature_71 = Pipeline([
    ('union', FeatureUnion(
        [
            ('text', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),
            ('URL', Pipeline([
                ('selector', ItemSelector(key='URL')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),
            ('userfollowers_count', Pipeline([
                ('selector', ItemSelector(key='userfollowers_count')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),
            ('favoriteCount', Pipeline([
                ('selector', ItemSelector(key='favoriteCount')),
                ('array', DataFrameToArrayTransformer()),
            ])),
            ('sentiment', Pipeline([
                ('selector', ItemSelector(key='sentiment')),
                ('array', DataFrameToArrayTransformer()),
            ])),
            ('source', Pipeline([
                ('selector', ItemSelector(key='source')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),
            ('neg', Pipeline([
                ('selector', ItemSelector(key='negative')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),

            ('pos', Pipeline([
                ('selector', ItemSelector(key='positive')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),

            ('textlen', Pipeline([
                ('selector', ItemSelector(key='textLen')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),
            ('processedtext', Pipeline([
                ('selector', ItemSelector(key='processedtext')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),

            ('entities', Pipeline([
                ('selector', ItemSelector(key='entities')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),

            ('spl', Pipeline([
                ('selector', ItemSelector(key='spl')),
                ('array', DataFrameToArrayTransformer()),
            ])),

            ('userLocation', Pipeline([
                ('selector', ItemSelector(key='userLocation')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ]))
        ])

     ),
    ('dummy_stratified', dummy_clf1)
])

pipeline_feature_72 = Pipeline([
    ('union', FeatureUnion(
        [
            ('text', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),

            ('entities', Pipeline([
                ('selector', ItemSelector(key='entities')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),

            ('URL', Pipeline([
                ('selector', ItemSelector(key='URL')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),
            ('userfollowers_count', Pipeline([
                ('selector', ItemSelector(key='userfollowers_count')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),
            ('favoriteCount', Pipeline([
                ('selector', ItemSelector(key='favoriteCount')),
                ('array', DataFrameToArrayTransformer()),
            ])),
            ('sentiment', Pipeline([
                ('selector', ItemSelector(key='sentiment')),
                ('array', DataFrameToArrayTransformer()),
            ])),
            ('source', Pipeline([
                ('selector', ItemSelector(key='source')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),
            ('neg', Pipeline([
                ('selector', ItemSelector(key='negative')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),
            ('pos', Pipeline([
                ('selector', ItemSelector(key='positive')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),

            ('textlen', Pipeline([
                ('selector', ItemSelector(key='textLen')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),
            ('processedtext', Pipeline([
                ('selector', ItemSelector(key='processedtext')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),

            ('spl', Pipeline([
                ('selector', ItemSelector(key='spl')),
                ('array', DataFrameToArrayTransformer()),
            ])),

            ('userLocation', Pipeline([
                ('selector', ItemSelector(key='userLocation')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ]))
        ])

     ),
    ('svc', SVC(gamma='auto'))
])

pipeline_feature_73 = Pipeline([
    ('union', FeatureUnion(
        [
            ('text', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),

            ('entities', Pipeline([
                ('selector', ItemSelector(key='entities')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),

            ('URL', Pipeline([
                ('selector', ItemSelector(key='URL')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),
            ('userfollowers_count', Pipeline([
                ('selector', ItemSelector(key='userfollowers_count')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),
            ('favoriteCount', Pipeline([
                ('selector', ItemSelector(key='favoriteCount')),
                ('array', DataFrameToArrayTransformer()),
            ])),
            ('sentiment', Pipeline([
                ('selector', ItemSelector(key='sentiment')),
                ('array', DataFrameToArrayTransformer()),
            ])),
            ('source', Pipeline([
                ('selector', ItemSelector(key='source')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),
            ('neg', Pipeline([
                ('selector', ItemSelector(key='negative')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),
            ('pos', Pipeline([
                ('selector', ItemSelector(key='positive')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),

            ('textlen', Pipeline([
                ('selector', ItemSelector(key='textLen')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),
            ('processedtext', Pipeline([
                ('selector', ItemSelector(key='processedtext')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),

            ('spl', Pipeline([
                ('selector', ItemSelector(key='spl')),
                ('array', DataFrameToArrayTransformer()),
            ])),

            ('userLocation', Pipeline([
                ('selector', ItemSelector(key='userLocation')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ]))
        ])

     ),
    ('dTree', DecisionTreeClassifier(random_state=10))
])

pipeline_feature_74 = Pipeline([
    ('union', FeatureUnion(
        [
            ('text', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),

            ('entities', Pipeline([
                ('selector', ItemSelector(key='entities')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),

            ('URL', Pipeline([
                ('selector', ItemSelector(key='URL')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),
            ('userfollowers_count', Pipeline([
                ('selector', ItemSelector(key='userfollowers_count')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),
            ('favoriteCount', Pipeline([
                ('selector', ItemSelector(key='favoriteCount')),
                ('array', DataFrameToArrayTransformer()),
            ])),
            ('sentiment', Pipeline([
                ('selector', ItemSelector(key='sentiment')),
                ('array', DataFrameToArrayTransformer()),
            ])),
            ('source', Pipeline([
                ('selector', ItemSelector(key='source')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),
            ('neg', Pipeline([
                ('selector', ItemSelector(key='negative')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),
            ('pos', Pipeline([
                ('selector', ItemSelector(key='positive')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),

            ('textlen', Pipeline([
                ('selector', ItemSelector(key='textLen')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),
            ('processedtext', Pipeline([
                ('selector', ItemSelector(key='processedtext')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),

            ('spl', Pipeline([
                ('selector', ItemSelector(key='spl')),
                ('array', DataFrameToArrayTransformer()),
            ])),

            ('userLocation', Pipeline([
                ('selector', ItemSelector(key='userLocation')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ]))
        ])

     ),
    ('rTree', RandomForestClassifier(random_state=10))
])

pipeline_feature_75 = Pipeline([
    ('union', FeatureUnion(
        [
            ('text', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),
            ('URL', Pipeline([
                ('selector', ItemSelector(key='URL')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),

            ('entities', Pipeline([
                ('selector', ItemSelector(key='entities')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),

            ('userfollowers_count', Pipeline([
                ('selector', ItemSelector(key='userfollowers_count')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),
            ('favoriteCount', Pipeline([
                ('selector', ItemSelector(key='favoriteCount')),
                ('array', DataFrameToArrayTransformer()),
            ])),
            ('sentiment', Pipeline([
                ('selector', ItemSelector(key='sentiment')),
                ('array', DataFrameToArrayTransformer()),
            ])),
            ('source', Pipeline([
                ('selector', ItemSelector(key='source')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),
            ('neg', Pipeline([
                ('selector', ItemSelector(key='negative')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),
            ('pos', Pipeline([
                ('selector', ItemSelector(key='positive')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),

            ('textlen', Pipeline([
                ('selector', ItemSelector(key='textLen')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),
            ('processedtext', Pipeline([
                ('selector', ItemSelector(key='processedtext')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),

            ('spl', Pipeline([
                ('selector', ItemSelector(key='spl')),
                ('array', DataFrameToArrayTransformer()),
            ])),

            ('userLocation', Pipeline([
                ('selector', ItemSelector(key='userLocation')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ]))
        ])

     ),
    ('Gaussian', BernoulliNB())
])

print("\nDummy classifier Before feature added \n")
pipeline_feature_51.fit(X_train22, y_train22)
resul151 = pipeline_feature_51.predict(X_val22)
evaluation_summary("Results with two feature on validation set", resul151, y_val22)
print("\nDummy classifier After feature added \n")
pipeline_feature_71.fit(X_train22, y_train22)
result71 = pipeline_feature_71.predict(X_val22)
evaluation_summary("Results with two feature on validation set", result71, y_val22)

print("\n SVC classifier Before feature added \n")
pipeline_feature_52.fit(X_train22, y_train22)
result152 = pipeline_feature_52.predict(X_val22)
evaluation_summary("Results with two feature on validation set", result152, y_val22)
print("\n SVC classifier After feature added \n")
pipeline_feature_72.fit(X_train22, y_train22)
result72 = pipeline_feature_72.predict(X_val22)
evaluation_summary("Results with two feature on validation set", result72, y_val22)

print("\n Decision tree classifier Before feature added \n")
pipeline_feature_53.fit(X_train22, y_train22)
result153 = pipeline_feature_53.predict(X_val22)
evaluation_summary("Results with two feature on validation set", result153, y_val22)
print("\n Decision tree classifier After feature added \n")
pipeline_feature_73.fit(X_train22, y_train22)
result73 = pipeline_feature_73.predict(X_val22)
evaluation_summary("Results with two feature on validation set", result73, y_val22)

print("\n RandomForest classifier Before feature added \n")
pipeline_feature_54.fit(X_train22, y_train22)
result154 = pipeline_feature_54.predict(X_val22)
evaluation_summary("Results with two feature on validation set", result154, y_val22)
print("\n RandomForest classifier After feature added \n")
pipeline_feature_74.fit(X_train22, y_train22)
result74 = pipeline_feature_74.predict(X_val22)
evaluation_summary("Results with two feature on validation set", result74, y_val22)

print("\n BernoulliNB classifier Before feature added \n")
pipeline_feature_55.fit(X_train22, y_train22)
result155 = pipeline_feature_55.predict(X_val22)
evaluation_summary("Results with two feature on validation set", result155, y_val22)
print("\n BernoulliNB classifier After feature added \n")
pipeline_feature_75.fit(X_train22, y_train22)
result75 = pipeline_feature_75.predict(X_val22)
evaluation_summary("Results with two feature on validation set", result75, y_val22)

"""# Summary after name entity"""

pipeline_feature_61 = Pipeline([
    ('union', FeatureUnion(
        [
            ('text', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),
            ('URL', Pipeline([
                ('selector', ItemSelector(key='URL')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),
            ('userfollowers_count', Pipeline([
                ('selector', ItemSelector(key='userfollowers_count')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),
            ('favoriteCount', Pipeline([
                ('selector', ItemSelector(key='favoriteCount')),
                ('array', DataFrameToArrayTransformer()),
            ])),
            ('sentiment', Pipeline([
                ('selector', ItemSelector(key='sentiment')),
                ('array', DataFrameToArrayTransformer()),
            ])),
            ('source', Pipeline([
                ('selector', ItemSelector(key='source')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),
            ('neg', Pipeline([
                ('selector', ItemSelector(key='negative')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),
            ('pos', Pipeline([
                ('selector', ItemSelector(key='positive')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),

            ('textlen', Pipeline([
                ('selector', ItemSelector(key='textLen')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),
            ('entities', Pipeline([
                ('selector', ItemSelector(key='entities')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),

            ('userLocation', Pipeline([
                ('selector', ItemSelector(key='userLocation')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ]))
        ])

     ),
    ('dummy_stratified', dummy_clf1)
])

pipeline_feature_62 = Pipeline([
    ('union', FeatureUnion(
        [
            ('text', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),
            ('URL', Pipeline([
                ('selector', ItemSelector(key='URL')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),
            ('userfollowers_count', Pipeline([
                ('selector', ItemSelector(key='userfollowers_count')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),
            ('favoriteCount', Pipeline([
                ('selector', ItemSelector(key='favoriteCount')),
                ('array', DataFrameToArrayTransformer()),
            ])),
            ('sentiment', Pipeline([
                ('selector', ItemSelector(key='sentiment')),
                ('array', DataFrameToArrayTransformer()),
            ])),
            ('source', Pipeline([
                ('selector', ItemSelector(key='source')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),
            ('neg', Pipeline([
                ('selector', ItemSelector(key='negative')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),
            ('pos', Pipeline([
                ('selector', ItemSelector(key='positive')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),

            ('textlen', Pipeline([
                ('selector', ItemSelector(key='textLen')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),

            ('entities', Pipeline([
                ('selector', ItemSelector(key='entities')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),

            ('userLocation', Pipeline([
                ('selector', ItemSelector(key='userLocation')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ]))
        ])

     ),
    ('svc', SVC(gamma='auto'))
])

pipeline_feature_63 = Pipeline([
    ('union', FeatureUnion(
        [
            ('text', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),
            ('URL', Pipeline([
                ('selector', ItemSelector(key='URL')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),
            ('userfollowers_count', Pipeline([
                ('selector', ItemSelector(key='userfollowers_count')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),
            ('favoriteCount', Pipeline([
                ('selector', ItemSelector(key='favoriteCount')),
                ('array', DataFrameToArrayTransformer()),
            ])),
            ('sentiment', Pipeline([
                ('selector', ItemSelector(key='sentiment')),
                ('array', DataFrameToArrayTransformer()),
            ])),
            ('source', Pipeline([
                ('selector', ItemSelector(key='source')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),
            ('neg', Pipeline([
                ('selector', ItemSelector(key='negative')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),
            ('pos', Pipeline([
                ('selector', ItemSelector(key='positive')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),

            ('textlen', Pipeline([
                ('selector', ItemSelector(key='textLen')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),

            ('entities', Pipeline([
                ('selector', ItemSelector(key='entities')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),

            ('userLocation', Pipeline([
                ('selector', ItemSelector(key='userLocation')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ]))
        ])

     ),
    ('dTree', DecisionTreeClassifier(random_state=10))
])

pipeline_feature_64 = Pipeline([
    ('union', FeatureUnion(
        [
            ('text', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),
            ('URL', Pipeline([
                ('selector', ItemSelector(key='URL')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),
            ('userfollowers_count', Pipeline([
                ('selector', ItemSelector(key='userfollowers_count')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),
            ('favoriteCount', Pipeline([
                ('selector', ItemSelector(key='favoriteCount')),
                ('array', DataFrameToArrayTransformer()),
            ])),
            ('sentiment', Pipeline([
                ('selector', ItemSelector(key='sentiment')),
                ('array', DataFrameToArrayTransformer()),
            ])),
            ('source', Pipeline([
                ('selector', ItemSelector(key='source')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),
            ('neg', Pipeline([
                ('selector', ItemSelector(key='negative')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),
            ('pos', Pipeline([
                ('selector', ItemSelector(key='positive')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),

            ('textlen', Pipeline([
                ('selector', ItemSelector(key='textLen')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),

            ('entities', Pipeline([
                ('selector', ItemSelector(key='entities')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),

            ('userLocation', Pipeline([
                ('selector', ItemSelector(key='userLocation')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ]))
        ])

     ),
    ('rTree', RandomForestClassifier(random_state=10))
])

pipeline_feature_65 = Pipeline([
    ('union', FeatureUnion(
        [
            ('text', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),
            ('URL', Pipeline([
                ('selector', ItemSelector(key='URL')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),
            ('userfollowers_count', Pipeline([
                ('selector', ItemSelector(key='userfollowers_count')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),
            ('favoriteCount', Pipeline([
                ('selector', ItemSelector(key='favoriteCount')),
                ('array', DataFrameToArrayTransformer()),
            ])),
            ('sentiment', Pipeline([
                ('selector', ItemSelector(key='sentiment')),
                ('array', DataFrameToArrayTransformer()),
            ])),
            ('source', Pipeline([
                ('selector', ItemSelector(key='source')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),
            ('neg', Pipeline([
                ('selector', ItemSelector(key='negative')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),
            ('pos', Pipeline([
                ('selector', ItemSelector(key='positive')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),

            ('textlen', Pipeline([
                ('selector', ItemSelector(key='textLen')),
                ('array', DataFrameToArrayTransformer()),
                # CountVectorizer(tokenizer=tokenize_normalize,binary=True,lowercase=False, max_features=20000)),
            ])),

            ('entities', Pipeline([
                ('selector', ItemSelector(key='entities')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ])),

            ('userLocation', Pipeline([
                ('selector', ItemSelector(key='userLocation')),
                ('one-hot',
                 CountVectorizer(tokenizer=tokenize_normalize, binary=True, lowercase=False, max_features=20000)),
            ]))
        ])

     ),
    ('Gaussian', BernoulliNB())
])

print("\nDummy classifier Before feature added \n")
pipeline_feature_71.fit(X_train22, y_train22)
result071 = pipeline_feature_71.predict(X_val22)
evaluation_summary("Results with two feature on validation set", result071, y_val22)
print("\nDummy classifier After feature added \n")
pipeline_feature_61.fit(X_train22, y_train22)
result61 = pipeline_feature_61.predict(X_val22)
evaluation_summary("Results with two feature on validation set", result61, y_val22)

print("\n SVC classifier Before feature added \n")
pipeline_feature_72.fit(X_train22, y_train22)
result072 = pipeline_feature_72.predict(X_val22)
evaluation_summary("Results with two feature on validation set", result072, y_val22)
print("\n SVC classifier After feature added \n")
pipeline_feature_62.fit(X_train22, y_train22)
result62 = pipeline_feature_62.predict(X_val22)
evaluation_summary("Results with two feature on validation set", result62, y_val22)

print("\n Decision tree classifier Before feature added \n")
pipeline_feature_73.fit(X_train22, y_train22)
result073 = pipeline_feature_73.predict(X_val22)
evaluation_summary("Results with two feature on validation set", result073, y_val22)
print("\n Decision tree classifier After feature added \n")
pipeline_feature_63.fit(X_train22, y_train22)
result63 = pipeline_feature_63.predict(X_val22)
evaluation_summary("Results with two feature on validation set", result63, y_val22)

print("\n RandomForest classifier Before feature added \n")
pipeline_feature_74.fit(X_train22, y_train22)
result074 = pipeline_feature_74.predict(X_val22)
evaluation_summary("Results with two feature on validation set", result074, y_val22)
print("\n RandomForest classifier After feature added \n")
pipeline_feature_64.fit(X_train22, y_train22)
result64 = pipeline_feature_64.predict(X_val22)
evaluation_summary("Results with two feature on validation set", result64, y_val22)

print("\n BernoulliNB classifier Before feature added \n")
pipeline_feature_75.fit(X_train22, y_train22)
result075 = pipeline_feature_75.predict(X_val22)
evaluation_summary("Results with two feature on validation set", result075, y_val22)
print("\n BernoulliNB classifier After feature added \n")
pipeline_feature_65.fit(X_train22, y_train22)
result65 = pipeline_feature_65.predict(X_val22)
evaluation_summary("Results with two feature on validation set", result65, y_val22)

#kfold = KFold(n_splits=5, random_state=42)
#cv = cross_val_score(pipeline_feature_74, data_frame, y, cv=kfold)

#cv.mean()

"""# Saving models"""

from sklearn.externals import joblib

train_set = firstcopy

joblib.dump(train_set, "./trainset.joblib", compress=True)
joblib.dump(pipeline_feature_71, "./dummy.joblib", compress=True)
joblib.dump(pipeline_feature_72, "./svc.joblib", compress=True)
joblib.dump(pipeline_feature_73, "./decisiontree.joblib", compress=True)
joblib.dump(pipeline_feature_74, "./randomforest.joblib", compress=True)
joblib.dump(pipeline_feature_75, "./bernoullinb.joblib", compress=True)

#modelReload_dummy = joblib.load("./dummy.joblib")
#modelReload_svc = joblib.load("./svc.joblib")
#modelReload_decision = joblib.load("./decisiontree.joblib")
#modelReload_randomforest = joblib.load("./randomforest.joblib")
#modelReload_bern = joblib.load("./bernoullinb.joblib")




