# YouTube Comment Spam Classifier

This project involves building a machine learning model to classify comments in YouTube videos as either ham (non-spam) or spam. The model is based on Natural Language Processing (NLP) techniques and utilizes data preprocessing, transformation, and the Multinomial Naive Bayes algorithm.

## Project Overview

- **Objective:**
  - Classify YouTube comments as ham or spam.

- **Techniques Used:**
  - Natural Language Processing (NLP)
  - Data Preprocessing
  - Data Transformation
  - CountVectorizer
  - TF-IDF Transformer
  - Multinomial Naive Bayes (MultinomialNB)

## Data

- The dataset used for training and testing the model consists of labeled YouTube comments, where each comment is classified as ham or spam.

## Workflow

1. **Data Preprocessing:**
   - Cleaned and preprocessed the raw text data, handling issues like special characters, punctuation, and irrelevant information.

2. **Data Transformation:**
   - Transformed the processed text data into a format suitable for machine learning models.

3. **Feature Extraction:**
   - Utilized CountVectorizer to convert the text data into a bag-of-words representation, capturing the frequency of words in each comment.

4. **TF-IDF Transformation:**
   - Applied TF-IDF (Term Frequency-Inverse Document Frequency) transformation to convert the count-based features into a numerical representation that considers the importance of each word in the entire dataset.

5. **Model Training:**
   - Trained the machine learning model using the Multinomial Naive Bayes algorithm.

6. **Evaluation:**
   - Evaluated the model's performance using relevant metrics such as accuracy, precision, recall, and F1-score.

## Dependencies

- The project is implemented in Python and relies on the following libraries:
  - scikit-learn
  - pandas
  - numpy
