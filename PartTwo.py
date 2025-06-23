import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
# Part - a
# i
# Read the CSV file from the 'texts' directory
df = pd.read_csv("p2-texts\p2-texts\hansard40000.csv")

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
# print(df['party'].value_counts())
# print(top_parties)

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
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
print(f"Train class distribution:\n{y_train.value_counts(normalize=True)}")


