import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import numpy as np
import re

# Part - a
# i
# Read the CSV file from the 'texts' directory
df = pd.read_csv("nlp-coursework-2024-25-ymogos\p2-texts\p2-texts\hansard40000.csv")

# Replace 'Labour (Co-op)' with 'Labour' in the 'party' column
df['party'] = df['party'].replace('Labour (Co-op)', 'Labour')

# a-ii
# Remove rows where 'party' is 'Speaker'
df = df[df['party'] != 'Speaker']

# Find the 4 most common party names (excluding 'Speaker')
top_parties = df['party'].value_counts().nlargest(4).index

# Keep only rows where 'party' is one of the top 4
df = df[df['party'].isin(top_parties)]

#  check result
#print(df['party'].value_counts())
#print(top_parties)

# iii
df = df[df['speech_class'] == 'Speech']

# iv
# Remove rows where 'speech' text is shorter than 1000 characters
df = df[df['speech'].str.len() >= 1000]

# Print the dimensions: (rows, columns)
#print(df.shape)

# Part - b

# Step 1: Initialize the vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)

# Step 2: Vectorise the 'speech' column
X = vectorizer.fit_transform(df['speech'])

# Step 3: Define the target variable (e.g., party classification)
y = df['party']

# Step 4: Split into train/test sets using stratified sampling
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,            # default 80% train / 20% test
    stratify=y,               # preserves class distribution
    random_state=26           # for reproducibility
)

# Step 5: Check result
#print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
#print(f"Train class distribution:\n{y_train.value_counts(normalize=True)}")

# Part - c (Classifiers)
def run_classifiers(X_train, X_test, y_train, y_test):
    # Random Forest
    rf = RandomForestClassifier(n_estimators=300, random_state=26)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    print(f"RandomForest Macro F1: {f1_score(y_test, y_pred_rf, average='macro'):.4f}")
    print(classification_report(y_test, y_pred_rf, zero_division=0))
    # SVM
    svm = SVC(kernel='linear', random_state=26)
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    print(f"SVM Macro F1: {f1_score(y_test, y_pred_svm, average='macro'):.4f}")
    print(classification_report(y_test, y_pred_svm, zero_division=0))
#run_classifiers(X_train, X_test, y_train, y_test)

# Part - d (N-gram Features)
tfidf_ngram = TfidfVectorizer(stop_words=None, max_features=3000, ngram_range=(1,3))
X_ngram = tfidf_ngram.fit_transform(df['speech'])
X_train_ng, X_test_ng, y_train_ng, y_test_ng = train_test_split(
    X_ngram, y, test_size=0.2, random_state=26, stratify=y)
#run_classifiers(X_train_ng, X_test_ng, y_train_ng, y_test_ng)

# Part - e (Custom Tokenizer)
def custom_tokenizer(text):
    # Example: simple lemmatization, remove numbers, keep words >2 chars
    tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    return tokens

tfidf_custom = TfidfVectorizer(tokenizer=custom_tokenizer, max_features=3000)
X_custom = tfidf_custom.fit_transform(df['speech'])
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_custom, y, test_size=0.2, random_state=26, stratify=y)

# Try both classifiers, print best
rf_c = RandomForestClassifier(n_estimators=300, random_state=26)
rf_c.fit(X_train_c, y_train_c)
y_pred_rf_c = rf_c.predict(X_test_c)
rf_f1 = f1_score(y_test_c, y_pred_rf_c, average='macro')

svm_c = SVC(kernel='linear', random_state=26)
svm_c.fit(X_train_c, y_train_c)
y_pred_svm_c = svm_c.predict(X_test_c)
svm_f1 = f1_score(y_test_c, y_pred_svm_c, average='macro')

if rf_f1 >= svm_f1:
    print("Custom Tokenizer - RandomForestClassifier")
    print(f"Macro F1: {rf_f1:.4f}")
    print(classification_report(y_test_c, y_pred_rf_c, zero_division=0))
    best_model = 'RandomForestClassifier'
    best_f1 = rf_f1
else:
    print("Custom Tokenizer - SVM")
    print(f"Macro F1: {svm_f1:.4f}")
    print(classification_report(y_test_c, y_pred_svm_c, zero_division=0))
    best_model = 'SVM'
    best_f1 = svm_f1

# Part - f (Explanation)
print("\nExplanation:")
print("""
The custom tokenizer lowercases the text, removes numbers and punctuation, and keeps only words with 
      at least 3 letters. This reduces noise and focuses on meaningful tokens, which can help the 
      classifier generalize better. The performance is reported above for the best classifier. 
      The trade-off is between capturing enough information for classification and keeping the feature
       space efficient (max 3000 features).
""")