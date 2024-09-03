# Harsh-comment-detection

## Overview

This project, titled **Why So Harsh**, aims to detect toxic comments and classify them into six mutually independent categories: "harsh," "extremely harsh," "vulgar," "threatening," "disrespect," and "targeted hate." We use a variety of Natural Language Processing (NLP) techniques and machine learning models to achieve this goal.

![Banner Image](https://github.com/shreyasic77/Harsh-comment-detection/blob/main/harsh_comment_image.jpeg)

## Project Structure

* **Notebook:** `harsh_comment_detection.ipynb` contains all the steps involved in data preprocessing, model building, training, and evaluation.
* **Report:** `Harsh_Comment_Detection_Project_Report.pdf` provides an in-depth explanation of the methodology, data analysis, and results.


## Project Details

### Objective

The main objective of this project is to build a model that can detect and classify new toxic comments into the following categories:

* Harsh
* Extremely Harsh
* Vulgar
* Threatening
* Disrespect
* Targeted Hate

### Data

* **Training Dataset:** 127,656 rows and 8 columns
* **Test Dataset:** 31,915 rows and 2 columns

The dataset is thoroughly analyzed for missing values, multi-tagging, and the distribution of toxic comments across different categories. Various visualization techniques like word clouds and correlation plots are used to gain insights.

### NLP Techniques

The project employs several NLP techniques for text preprocessing and feature extraction:

* **Text Preprocessing:** Removal of special characters, digits, stop words; tokenization; lemmatization; and abbreviation expansion.
* **Feature Engineering:**
  * Word2Vec (Skip-gram and CBOW models)
  * Sentence Vectorization
  * CountVectorizer
  * TF-IDF Vectorization (Unigrams, Bigrams, and Trigrams)

### Modeling

Multiple machine learning models are tested, including:

* Logistic Regression
* Multinomial Naive Bayes
* SGD Classifier

The models are evaluated using metrics like ROC-AUC, F1-score, precision, and recall. Feature extraction methods, such as the count of harsh words or threat words, are used to improve model accuracy.

### Results

* **Best Model:** Logistic Regression with TF-IDF Vectorization.
* **Accuracy:** The model achieved a Kaggle score of 0.98276 on the private leaderboard.
* **Other Notable Models:**
  * SGD Classifier with Squared Hinge Loss (Kaggle score: 0.98218)

### Conclusion
In conclusion, the Logistic Regression model with TF-IDF vectorization and the SGD Classifier with Squared Hinge Loss are the most effective for harsh comment detection. Their high accuracy and robust performance across multiple categories make them suitable for real-world applications.

These models were highly effective in classifying toxic comments across different categories.

  
