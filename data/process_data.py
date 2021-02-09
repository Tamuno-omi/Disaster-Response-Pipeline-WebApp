import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

CATEGORIES_FILENAME = 'categories.csv'
MESSAGES_FILENAME = 'messages.csv'
DATABASE_FILENAME = '../db.sqlite3'
TABLE_NAME = 'disaster_message'



def load_data(messages_filepath, categories_filepath):
    '''load and merges the message and categories datasets.
    inputs
    message_filepath: messages dataset
    categories_filepath: categories dataset
    
    output 
    df: merged message & categories dataset.    
    '''
    # read in messages dataset
    messages = pd.read_csv(messages_filepath)
    
    #read in categories dataset
    categories = pd.read_csv(categories_filepath)
    #merge messages & categories datasets by id
    df = pd.merge(messages, categories, on = 'id')
    
    return df
    
def clean_data(df):
    '''cleans data by splitting categories into different colums, create and rename columns with first row of categories & dropping duplicates
  input
  df: merged messages & categories
  
  output
  df: cleaned df
    '''
     # create a dataframe of the 36 individual category columns# select the first row of the categories  dataframe
    categories = df.categories.str.split(pat = ';', expand = True)
    # select the first row of the categories dataframe
    row = categories.iloc[0,:]
    #apply a lambda function that takes everything up to the second to last character of each string with  slicing
    category_colnames = row.apply(lambda x:x[:-2])
    # rename the columns of `categories`
    categories.columns = category_colnames
    # convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    # drop the original categories column from `df`
    df = df.drop('categories', axis = 1)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)
    # drop duplicates
    df = df.drop_duplicates()
    df['related'].replace([2], [1], inplace=True)
    return df
    


def save_data(df, database_filename):
    '''saves dataframe to sql databases
    input
    
    df: cleaned dataframe
    database_filename: the name of the database file
    
    '''
    engine = create_engine('sqlite:///DisasterResponse.db')
    df.to_sql('Disasters', engine, index=False, if_exists='replace')

    
def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()  