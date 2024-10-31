# NLP Sentiment Analysis on Twitter Data

## Overview
This repository contains the code and documentation for my final year, final semester project in Natural Language Processing (NLP). This project aims to classify the sentiment of tweets as either positive or negative, implemented using four machine learning and deep learning models: SVM, BERT, Logistic Regression with TF-IDF, and LSTM. Each model is evaluated for performance, allowing insights into their efficacy on sentiment analysis for social media data.

## Table of Contents
- Project Goals
- Dataset
- Models Implemented
- Predicting Sentiment
- Results and Visualization
- Comparative Performance

## Project Goals
- Develop and compare multiple NLP models for sentiment analysis of Twitter data.
- Analyze the performance of each model in terms of accuracy, precision, recall, and F1-score.
- Gain insights into the effectiveness of different NLP techniques for real-time social media sentiment analysis.

## Dataset
The dataset used for this project is a subset of the popular Twitter sentiment analysis dataset. It contains tweets labeled as either positive or negative:

- Positive tweets labeled with 4.
- Negative tweets labeled with 0.

For model evaluation, the data is split into training and test sets, with metrics collected to assess model performance.

## Models Implemented
This project includes four different models to predict tweet sentiment:

- Logistic Regression with TF-IDF: A baseline model that uses TF-IDF for text vectorization, followed by a logistic regression classifier.
- Support Vector Machine (SVM): Using a linear SVM with TF-IDF features, optimized for binary sentiment classification.
- BERT: A pre-trained transformer model fine-tuned for the sentiment analysis task.
- LSTM: A Recurrent Neural Network (RNN) model that uses tokenized and padded tweet sequences for sentiment classification.

## Predicting Sentiment
Each model includes a function for predicting sentiment from new input. For example, in BERT:
- Enter a tweet to predict its sentiment: he is to bad in this subject
- The predicted sentiment for the tweet is: Negative

## Results and Visualizations
The repository includes code to visualize model performance:

- Loss Over Epochs: Tracks how each model learns over time.
- Confusion Matrix: Displays true vs. predicted labels for test data.
- Classification Metrics: Bar charts of precision, recall, and F1-score for positive and negative classes.

## Comparative Performance
### Accuracy:
- Logistic Regression with TF-IDF: 79%
- SVM: 71%
- BERT: 77%
- LSTM: 66%

