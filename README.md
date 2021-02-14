# Disaster Response Pipeline Project

## Table of Contents
* [Introduction](#introduction)
* [Installations](#installations)
* [File Descriptions](#file-descriptions)
* [Instructions](#instructions)
* [Results](#results)
  - [Classifier](#classifier)
  - [Web App](#web-app)
* [Acknowledgements](#acknowledgements)

## Introduction
In this project a web application was created to classify disaster messages into appropriate categories. The project has three key components:

1) ETL Pipeline : Loads the necessary datasets, cleans and stores it in an Sqlite database.

2) ML Pipeline: A python script that loads the Sqlite database, splits it into training and test sets, processes texts, trains and tunes the model then output the results.

3) Flask Web App: Displays visualizations & interface for typing messages and displaying classifications.

## Installations
The following packages were used for this project and can be found in the standard Anaconda distribution for Python 3.7:
* Jupyter notebook
* NumPy
* Pandas
* Scikit-learn
* NLTK
* Flask
* Plotly

## File Descriptions
There are four directories in this repository:
#### data
* `disaster_messages.csv`: CSV file containing over  messages sent during natural disasters.
* `disaster_categories.csv`: CSV file containing 36 category labels for the messages in `disaster_messages.csv`.
* `process_data.py`: Python script to run the ETL pipeline.  
* `DisasterResponse.db`: SQLite database containing the merged, transformed and cleaned data that is ready for the machine learning pipeline

#### models
* `train_classifier.py`: Python script to run the ML pipeline that builds the model, trains it, tunes hyperparameters using GridSearch, displays the model performance results on the test data, and saves the best model to a pickle file.
* `model.pkl`: pickle file containing trained model.

#### app
* `run.py`: Python script to run the web app displaying visualizations of the dataset and allowing users to input a message and view its categories.
* `templates`: directory containing 2 html files:
  -  `master.html`: Main page of web app. Contains visualizations of messages dataset.
  -  `go.html`: Classification result page of web app.

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
