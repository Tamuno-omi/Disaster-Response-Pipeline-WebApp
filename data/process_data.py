import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''load and merges the message and categories datasets.
    inputs
    message_filepath: messages dataset
    categories_filepath: categories dataset
    
    output 
    df: merged message & categories dataset.    
    '''
    #read in messages dataset
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
    categories =  df['categories'].str.split(pat = ';', expand = True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]

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
    df = df.drop('categories', axis =1)
        
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
        
    # drop duplicates
    df = df.drop_duplicates()
    return df

def save_data(df, database_filename, table_name):
    '''saves dataframe to sql databases
    input
    
    df: cleaned dataframe
    database_filename: the name of the database file
    table_name: name for table in database file
    
    '''
    # save dataset to sqlite database 
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql(table_name, engine, index=False, if_exists='replace')

def main():
    # if no table name is given by the user, save data to 'merged'
    if len(sys.argv) == 4:

        # assign variables from system arguments
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        table_name = 'merged'

        # run load_data
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        # run clean_data
        print('Cleaning data...')
        df = clean_data(df)

        # run save_data
        print('Saving data...\n    DATABASE: {}\n    TABLE: {}'.format(database_filepath, table_name))
        save_data(df, database_filepath, table_name)

        print('Cleaned data saved to database!')

    # if table name is given by the user, save data to table_name
    elif len(sys.argv) == 5:

        # assign variables from system arguments
        messages_filepath, categories_filepath, database_filepath, table_name = sys.argv[1:]

        # run load_data
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        # run clean_data
        print('Cleaning data...')
        df = clean_data(df)

        # run save_data
        print('Saving data...\n    DATABASE: {}\n    TABLE: {}'.format(database_filepath, table_name))
        save_data(df, database_filepath, table_name)

        print('Cleaned data saved to database!')

    # if the incorrect number of inputs is give, ask for the correct number
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. Optional: provide the table name to '\
              'save to as well.\n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db (optional)merged')


if __name__ == '__main__':
    main()