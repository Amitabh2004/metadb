def naivebayes():
    print("""
import pandas as pd
import string
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Step 1: Load the SMS Spam dataset (tab-separated, no headers)
df = pd.read_csv("SMSSpamCollection", sep='\\t', header=None, names=["label", "text"])

# Preprocess text: lowercase, remove punctuation, remove stopwords
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    filtered = [word for word in tokens if word not in stopwords.words('english')]
    return " ".join(filtered)

df['clean_text'] = df['text'].apply(preprocess)

# Step 2: Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['label'], test_size=0.2, random_state=42)

# Step 3: Train Multinomial Naive Bayes
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Step 4: Evaluation
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, pos_label='spam'))
print("Recall:", recall_score(y_test, y_pred, pos_label='spam'))
print("F1-Score:", f1_score(y_test, y_pred, pos_label='spam'))

# Step 5: Prediction on sample message
sample = ["Win a free iPhone now!"]
sample_clean = [preprocess(sample[0])]
sample_vec = vectorizer.transform(sample_clean)
print("Prediction for sample message:", model.predict(sample_vec)[0])

# Step 6: Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=['ham', 'spam'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Ham', 'Spam'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()   
        """)