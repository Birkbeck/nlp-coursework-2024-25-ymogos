import pandas as pd
# a-i)
# Read the CSV file from the 'texts' directory
df = pd.read_csv("p2-texts\p2-texts\hansard40000.csv")

# Replace 'Labour (Co-op)' with 'Labour' in the 'party' column
df['party'] = df['party'].replace('Labour (Co-op)', 'Labour')

# a-ii)
# Remove rows where 'party' is 'Speaker'
df = df[df['party'] != 'Speaker']

# Find the 4 most common party names (excluding 'Speaker')
top_parties = df['party'].value_counts().nlargest(4).index

# Keep only rows where 'party' is one of the top 4
df = df[df['party'].isin(top_parties)]

#  check result
# print(df['party'].value_counts())
# print(top_parties)

# a-iii)
df = df[df['speech_class'] == 'Speech']

# a-iv)
# Remove rows where 'speech' text is shorter than 1000 characters
df = df[df['speech'].str.len() >= 1000]

# Print the dimensions: (rows, columns)
print(df.shape)


