import pandas as pd

# Read the CSV file from the 'texts' directory
df = pd.read_csv("p2-texts\p2-texts\hansard40000.csv")

# Replace 'Labour (Co-op)' with 'Labour' in the 'party' column
df['party'] = df['party'].replace('Labour (Co-op)', 'Labour')

# Display to confirm change
print(df['party'].value_counts())  # Optional: check if change occurred
