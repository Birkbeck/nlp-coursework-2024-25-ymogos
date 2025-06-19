
#Re-assessment template 2025

# Note: The template functions here and the dataframe format for structuring your solution is a suggested but not mandatory approach. You can use a different approach if you like, as long as you clearly answer the questions and communicate your answers clearly.

import nltk
import spacy
from pathlib import Path
import pandas as pd
from collections import Counter
import math


nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000



def fk_level(text, d):
    """Returns the Flesch-Kincaid Grade Level of a text (higher grade is more difficult).
    Requires a dictionary of syllables per word.

    Args:
        text (str): The text to analyze.
        d (dict): A dictionary of syllables per word.

    Returns:
        float: The Flesch-Kincaid Grade Level of the text.
    """
    # Tokenize into sentences and words
    sentences = nltk.sent_tokenize(text)
    words = nltk.word_tokenize(text)

    # Count total syllables using dictionary
    def count_syllables(word):
        word = word.lower()
        if word in d:
            # Count phonemes that end with a digit (stressed syllables)
            return len([ph for ph in d[word][0] if ph[-1].isdigit()])
        # Estimate syllables if word not in dictionary
        import re
        return max(1, len(re.findall(r'[aeiouy]+', word)))

    syllables = sum(count_syllables(word) for word in words if word.isalpha())

    num_sentences = len(sentences)
    num_words = len([word for word in words if word.isalpha()])

    if num_sentences == 0 or num_words == 0:
        return 0.0

    fk_grade = 0.39 * (num_words / num_sentences) + 11.8 * (syllables / num_words) - 15.59
    return round(fk_grade, 2)

def count_syl(word, d):
    """Counts the number of syllables in a word given a dictionary of syllables per word.
    if the word is not in the dictionary, syllables are estimated by counting vowel clusters

    Args:
        word (str): The word to count syllables for.
        d (dict): A dictionary of syllables per word.

    Returns:
        int: The number of syllables in the word.
    """
    word = word.lower()
    if word in d:
        # CMU dict: list of pronunciations, each is a list of phonemes
        return len([ph for ph in d[word][0] if ph[-1].isdigit()])
    # Estimate: count contiguous vowel groups
    import re
    return max(1, len(re.findall(r'[aeiouy]+', word)))



def read_novels(path=Path.cwd() / "texts" / "novels"):
    """Reads texts from a directory of .txt files and returns a DataFrame with the text, title,
    author, and year. The DataFrame is sorted by year (as integer) and index is reset."""
    data = []
    path = Path(path)
    for file in path.glob("*.txt"):
        # Extract metadata from filename: Title-Author-Year.txt
        name = file.stem  # Remove .txt
        try:
            # Split on last two dashes for author and year
            parts = name.rsplit("-", 2)
            if len(parts) == 3:
                title, author, year = parts
            else:
                title, author, year = parts[0], "Unknown", "Unknown"
        except Exception:
            title, author, year = name, "Unknown", "Unknown"
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()
        data.append({
            "text": text,
            "title": title.replace("_", " "),
            "author": author.replace("_", " "),
            "year": year
        })
    df = pd.DataFrame(data, columns=['text', 'title', 'author', 'year'])
   
    # Convert year to int for sorting, handle non-integer years gracefully
    def to_int(val):
        try:
            return int(val)
        except:
            return float('inf')
    df['year'] = df['year'].apply(to_int)
    df = df.sort_values('year').reset_index(drop=True)
    return df

def parse(df, store_path=Path.cwd() / "pickles", out_name="parsed.pickle"):
    """Parses the text of a DataFrame using spaCy, stores the parsed docs as a column and writes 
    the resulting  DataFrame to a pickle file"""
    store_path = Path(store_path)
    store_path.mkdir(parents=True, exist_ok=True)
    parsed_docs = []
    for i, text in enumerate(df["text"]):
        print(f"Parsing text {i+1}/{len(df)}: {df['title'].iloc[i]}")
        # Handle long texts by splitting if needed
        if len(text) > nlp.max_length:
            docs = []
            for j in range(0, len(text), nlp.max_length):
                docs.append(nlp(text[j:j+nlp.max_length]))
            # Concatenate Doc objects
            from spacy.tokens import Doc
            doc = Doc.from_docs(docs)
        else:
            doc = nlp(text)
        parsed_docs.append(doc)
    df = df.copy()
    df["parsed"] = parsed_docs
    out_path = store_path / out_name
    df.to_pickle(out_path)
    return df



def nltk_ttr(text):
    """Calculates the type-token ratio of a text. Text is tokenized using nltk.word_tokenize. Ignores punctuation and is case-insensitive."""
    tokens = [word.lower() for word in nltk.word_tokenize(text) if word.isalpha()]
    if not tokens:
        return 0.0
    types = set(tokens)
    return round(len(types) / len(tokens), 4)

def get_ttrs(df):
    """helper function to add ttr to a dataframe"""
    results = {}
    for i, row in df.iterrows():
        results[row["title"]] = nltk_ttr(row["text"])
    return results


def get_fks(df):
    """helper function to add fk scores to a dataframe"""
    results = {}
    cmudict = nltk.corpus.cmudict.dict()
    for i, row in df.iterrows():
        results[row["title"]] = round(fk_level(row["text"], cmudict), 4)
    return results


def subjects_by_verb_pmi(doc, target_verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    target_verb = target_verb.lower()
    subjects = []
    all_subjects = []
    all_verbs = []
    for token in doc:
        if token.pos_ == "VERB":
            all_verbs.append(token.lemma_.lower())
            for child in token.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    all_subjects.append(child.lemma_.lower())
                    if token.lemma_.lower() == target_verb:
                        subjects.append(child.lemma_.lower())
    subj_counts = Counter(all_subjects)
    verb_count = all_verbs.count(target_verb)
    subj_verb_counts = Counter(subjects)
    total_verbs = len(all_verbs)
    total_subjects = len(all_subjects)
    pmi_scores = {}
    for subj, joint in subj_verb_counts.items():
        p_subj = subj_counts[subj] / total_subjects if total_subjects else 0
        p_verb = verb_count / total_verbs if total_verbs else 0
        p_joint = joint / verb_count if verb_count else 0
        if p_subj > 0 and p_verb > 0 and p_joint > 0:
            pmi_scores[subj] = math.log2(p_joint / (p_subj * p_verb))
        else:
            pmi_scores[subj] = float('-inf')
    # Return top 10 by PMI
    return sorted(pmi_scores.items(), key=lambda x: x[1], reverse=True)[:10]




def subjects_by_verb_count(doc, verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    from collections import Counter
    subjects = []
    verb = verb.lower()
    for token in doc:
        if token.pos_ == "VERB" and token.lemma_.lower() == verb:
            for child in token.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    subjects.append(child.lemma_.lower())
    return Counter(subjects).most_common(10)


def adjective_counts(doc):
    """Extracts the most common adjectives in a parsed document. Returns a list of tuples."""
    pass



if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """
    #path = Path.cwd() / "p1-texts" / "novels"
    #print(path)
    #df = read_novels(path) # this line will fail until you have completed the read_novels function above.
    #print(df.head())
    #nltk.download("cmudict")
    #parse(df)
    #print(df.head())
    #print(get_ttrs(df))
    #print(get_fks(df))
    #df = pd.read_pickle(Path.cwd() / "pickles" /"name.pickle")
    # print(adjective_counts(df))
    """ 
    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_count(row["parsed"], "hear"))
        print("\n")

    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_pmi(row["parsed"], "hear"))
        print("\n")
    """

