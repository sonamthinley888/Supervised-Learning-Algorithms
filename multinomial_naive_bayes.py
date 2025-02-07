import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Download necessary NLTK resources
nltk.download('stopwords')

# Load the dataset
data = pd.read_csv('data.csv')

# Prepare the data
X = data['text']
Y = data['sentiment']

# Split into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Vectorize the text data
# vectorizer = CountVectorizer(stop_words=nltk.corpus.stopwords.words('english'))

vectorizer = TfidfVectorizer(stop_words=nltk.corpus.stopwords.words('english'))

# vectorizer = TfidfVectorizer(
#     stop_words=nltk.corpus.stopwords.words('english'),
#     max_features=1000,         # Limit to top 1000 features
#     ngram_range=(1, 2),        # Use unigrams and bigrams
#     sublinear_tf=True          # Use sublinear term frequency scaling
# )


X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train the model 
model = MultinomialNB()

model.fit(X_train_vectorized, Y_train)

# Test the model
y_pred = model.predict(X_test_vectorized)

# Check Accuracy
print(f'Accuracy: {accuracy_score(Y_test, y_pred)}')


# Function to predict the sentiment of a new text
def predict_sentiment(text):
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)
    return prediction[0]


# Example usage 
if __name__ == "__main__":
    new_text = "I hate this product"
    print(f'Sentiment: {predict_sentiment(new_text)}')


print(f"Predictions: {y_pred}")
print(f"Actual: {Y_test.tolist()}")
print(f"Model name: {model}")
