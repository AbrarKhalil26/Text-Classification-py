import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn import metrics
import random

def main():
    try:
        input_file = input("Enter the input file name: ")
        lang = input("Choose the language (1. English / 2. Arabic): ")
        if lang == "1":
            test(input_file, lang="English")
        elif lang == "2":
            test(input_file, lang="Arabic")
        else:
            print("Invalid input")
            sys.exit(1)
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)
    except PermissionError:
        print(f"Error: Permission denied for file '{input_file}'.")
        sys.exit(1)

def test(input_file, lang):
    df = pd.read_csv(input_file, sep='\t')
    df.dropna(inplace=True)
    
    X = df['review']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    text_clf = Pipeline([('tfidf', TfidfVectorizer())])
    if lang == "English":
        classifier = random.choice([MultinomialNB(), LinearSVC()])
        text_clf.steps.append(('clf', classifier))
    elif lang == "Arabic":
        classifier = LinearSVC()
        text_clf.steps.append(('clf', classifier))
        df_shuffled = df.sample(frac=1, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(df_shuffled['review'], df_shuffled['label'], test_size=0.33, random_state=42)
    
    text_clf.fit(X_train, y_train)
    predictions = text_clf.predict(X_test)
    print(f"Classification Report ({lang}):")
    print(metrics.classification_report(y_test, predictions))

if __name__ == "__main__":
    main()


# moviereviews.tsv
# ar_reviews_100k.tsv