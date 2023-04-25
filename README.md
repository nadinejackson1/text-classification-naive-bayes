### 100 Days of Machine Learning - Day 17

# Naive Bayes Text Classifier

This project demonstrates how to implement a Naive Bayes algorithm for text classification using Python and scikit-learn. The classifier categorizes social media posts, news articles, or NGO reports into categories such as human rights or sustainability, etc.

### Dataset

The dataset used in this project is the "Categorized News Articles" dataset available on Kaggle. You can download the dataset [here](https://www.kaggle.com/rmisra/news-category-dataset).

### Requirements

Install the following Python libraries:

- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn

You can install them using pip:

    pip install numpy pandas scikit-learn matplotlib seaborn

### Implementation Steps

1. Load the dataset using pandas and explore its structure
2. Preprocess the dataset by combining the title and short_description columns, filtering relevant categories, and encoding categories into numerical labels
3. Split the dataset into training and testing sets
4. Vectorize the text data using the Bag of Words model with TF-IDF weighting
5. Train the Naive Bayes classifier using the training data
6. Evaluate the classifier using accuracy, precision, recall, and F1-score
7. Visualize the results using a confusion matrix

### Usage

- Download the dataset from Kaggle and place it in the project directory
- Run the Python script containing the implementation to train and evaluate the classifier

### Improving the Classifier

To improve the classifier's performance, consider the following:

- Use a more complex model, such as Logistic Regression, Support Vector Machines, or Deep Learning-based models like BERT
- Resample the dataset to balance the categories, either by oversampling the minority class or undersampling the majority class
- Perform more advanced preprocessing, like stemming or lemmatization, to reduce the dimensionality of the text features
- Use other feature extraction techniques, such as word embeddings (e.g., Word2Vec, GloVe) or contextual embeddings (e.g., BERT, RoBERTa)