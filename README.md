SMS Spam Classifier using Machine Learning and NLP
A machine learning project that classifies SMS messages as Spam or Ham (Not Spam) using Natural Language Processing (NLP). The project explores message patterns, preprocesses text, compares multiple classification algorithms, evaluates model performance, and saves a trained model and TF-IDF vectorizer for future predictions.

Project Overview
Spam messages often contain unwanted advertisements, suspicious links, fake prizes, or fraudulent offers. This project uses text classification to identify whether a message is spam or a legitimate message.

The notebook follows a typical machine learning workflow:

Load and inspect the SMS dataset.
Clean the dataset and remove duplicate messages.
Explore the distribution and characteristics of spam and ham messages.
Preprocess message text using tokenization, stop-word removal, and stemming.
Convert text into numerical features using CountVectorizer and TfidfVectorizer.
Train and compare several machine learning classifiers.
Evaluate predictions using accuracy, precision, recall, F1-score, and a confusion matrix.
Save the selected classifier and TF-IDF vectorizer with Pickle.
Features
SMS spam and ham classification
Data cleaning and duplicate removal
Exploratory data analysis and visualizations
Message-length, word-count, and sentence-count analysis
Text normalization and preprocessing
Word clouds and frequent-word analysis
Bag of Words and TF-IDF feature extraction
Comparison of multiple machine learning algorithms
Model evaluation with classification metrics
Prediction function for new messages
Export of the trained model and vectorizer
Technologies Used
Python
Pandas and NumPy — data handling
Matplotlib and Seaborn — visualizations
NLTK — tokenization, stop words, and stemming
WordCloud — word-cloud visualizations
Scikit-learn — vectorization, model training, and evaluation
XGBoost — additional classifier comparison
Pickle — saving trained objects
Jupyter Notebook — development environment
Project Structure
sms-spam-classifier/
├── spam_classifier.ipynb
├── spam.csv
├── spam_classifier.pkl
├── tfidf_vectorizer.pkl
└── README.md
spam_classifier.ipynb — notebook containing data analysis, preprocessing, model comparison, evaluation, and prediction.
spam.csv — input SMS dataset expected by the notebook.
spam_classifier.pkl — saved Bernoulli Naive Bayes classifier.
tfidf_vectorizer.pkl — saved TF-IDF vectorizer.
README.md — project documentation.
The two .pkl files are created when the notebook's final saving cell is executed. The dataset is not included automatically; provide the dataset separately if you share the project.

Dataset
The notebook loads the dataset using:

df = pd.read_csv("spam.csv", encoding="latin-1")
It renames the relevant columns to:

target — message label (ham or spam, encoded as numeric values)
text — SMS message content
The notebook removes extra unnamed columns and duplicate rows before analysis. Place spam.csv in the working directory used to run the notebook, or update the file path accordingly.

Installation and Setup
1. Clone the repository
git clone <your-repository-url>
cd <your-repository-folder>
Replace the placeholders with your GitHub repository URL and folder name.

2. Create and activate a virtual environment (optional)
python -m venv .venv
On Windows:

.venv\Scripts\activate
On macOS or Linux:

source .venv/bin/activate
3. Install dependencies
pip install numpy pandas matplotlib seaborn nltk wordcloud scikit-learn xgboost jupyter
4. Start Jupyter Notebook
jupyter notebook
Open spam_classifier.ipynb and run the cells in order. The notebook downloads the required NLTK resources (punkt, punkt_tab, and stopwords), so an internet connection may be needed the first time these resources are downloaded.

Text Preprocessing
The notebook defines a transform_text() function that:

Converts text to lowercase.
Tokenizes the message.
Keeps alphanumeric tokens.
Removes English stop words and punctuation.
Applies Porter stemming.
Joins the processed tokens into a single string.
This produces a normalized text representation that can be passed to the vectorizer.

Feature Extraction
The project compares two ways of converting text into numerical features:

CountVectorizer (Bag of Words): represents messages using word-occurrence counts.
TfidfVectorizer (TF-IDF): represents messages using term-frequency and inverse-document-frequency weights.
The vectorizers are fitted on the training text and then used to transform the test text, helping prevent information from the test set leaking into feature extraction.

Models Compared
The notebook evaluates several classifiers with both CountVectorizer and TF-IDF features. These include:

Gaussian Naive Bayes
Multinomial Naive Bayes
Bernoulli Naive Bayes
Support Vector Classifier (SVC)
K-Nearest Neighbors
Decision Tree
Logistic Regression
Random Forest
AdaBoost
Bagging
Extra Trees
Gradient Boosting
XGBoost
The notebook selects Bernoulli Naive Bayes with TF-IDF features for its final prediction function and saves that classifier alongside the fitted TF-IDF vectorizer.

Evaluation Metrics
The notebook uses the following metrics to assess classification performance:

Accuracy: proportion of all predictions that are correct.
Precision: proportion of messages predicted as spam that are actually spam.
Recall: proportion of actual spam messages that are correctly detected.
F1-score: harmonic mean of precision and recall.
Confusion matrix: counts true negatives, false positives, false negatives, and true positives.
The notebook also compares accuracy and precision across classifiers and feature-extraction methods. Actual metric values depend on the dataset and the execution environment; refer to the notebook's output for the results from your run.

Predict a New Message
After running the notebook cells that define transform_text(), fit tfidf, and train bnb, use:

predict_message("Congratulations! You have won a free prize. Claim now!")
The function returns either Spam or Not Spam (Ham).

Load the Saved Model
After the notebook has created spam_classifier.pkl and tfidf_vectorizer.pkl, you can load them in another Python script:

import pickle

with open("spam_classifier.pkl", "rb") as file:
    model = pickle.load(file)

with open("tfidf_vectorizer.pkl", "rb") as file:
    tfidf = pickle.load(file)
To predict a message outside the notebook, you must also reuse the same text preprocessing function used during training:

message = "Congratulations! You have won a free prize. Claim now!"

processed_message = transform_text(message)
message_vector = tfidf.transform([processed_message])
prediction = model.predict(message_vector)[0]

print("Spam" if prediction == 1 else "Not Spam (Ham)")
Make sure transform_text() is defined in your script, along with the required NLTK, string, and Porter stemmer imports. Only load Pickle files from sources you trust.

Important Notes
Run the notebook from a directory where spam.csv is available, or change the dataset path.
Run notebook cells in order because later cells depend on objects created earlier.
The saved classifier and vectorizer must be used together with the same preprocessing steps.
The notebook converts sparse feature matrices to dense arrays during its model-comparison loops. For larger datasets, this can consume substantial memory; use sparse-compatible classifiers or revise the comparison code if needed.
This is a learning project. Predictions may be incorrect, and the model should not be treated as a guarantee that a message is safe or fraudulent.



Learning Outcomes
Through this project, I practised:

Cleaning and exploring a text dataset with Pandas
Performing NLP preprocessing with NLTK
Visualizing message and word-frequency patterns
Converting text into numerical features
Training and comparing classification algorithms
Evaluating classification results
Saving and reusing a trained machine learning model
Author
Arpan Dawn

If you found this project useful, feel free to star the repository or share suggestions for improvement.

