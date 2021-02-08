import sys
import os
import re

from sqlalchemy import create_engine
import pickle
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 500)
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from scipy.stats import gmean
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.metrics import classification_report, accuracy_score


def load_data(database_filepath):
    '''Function to load in sqlite database from file
    Input Parameters
    database_filepath: path to sqlite database
    
    Returns
    X: Messages column(feature)
    Y: Categories columns(dependent variable)
    category_names: names of categories
    '''
    engine = create_engine('sqlite:///DisasterResponse.db')
    df = pd.read_sql_table('Disasters', engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = list(df.columns[4:])
    return X, Y, category_names


def tokenize(text):
    """ tokenize function for processing text data
    
        Input parameters
        text : str, text to tokenize
    
        Returns
        clean_tokens: list, list of tokens
        """   
    # remove punctutation marks
    text = re.sub(r'[^a-zA-Z0-9]',' ',text)
    
    # break (text) into individual linguistic units
    tokens = word_tokenize(text)
    
    # lemmatize to remove inflections and derivations from a word
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()