import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn import metrics

def main():
    try:
        input_file = input("Enter the input file name: ")
        test(input_file)
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)
    except PermissionError:
        print(f"Error: Permission denied for file '{input_file}'.")
        sys.exit(1)

def test(input_file):
    df = pd.read_csv(input_file, sep='\t')
    df.isnull().sum()
    df.dropna(inplace=True)

    blanks = []
    for i, label, reviews in df.itertuples():
        if type(reviews) == str:
            if reviews.isspace():
                blanks.append(i)

    print(len(blanks), 'blanks: ', blanks)
    df.drop(blanks, inplace=True)
    
    X = df['review']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    text_clf_nb = Pipeline([('tfidf', TfidfVectorizer()), ('clf', MultinomialNB())])
    text_clf_nb.fit(X_train, y_train)
    predictions = text_clf_nb.predict(X_test)

    print(f"Accuracy : {metrics.accuracy_score(y_test, predictions)}")
    print("Classification Report :")
    print(metrics.classification_report(y_test, predictions))

    text_clf_lsvc = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LinearSVC())])
    text_clf_lsvc.fit(X_train, y_train)
    predictions = text_clf_lsvc.predict(X_test)

    print(f"Accuracy : {metrics.accuracy_score(y_test, predictions)}")
    print("Classification Report :")
    print(metrics.classification_report(y_test, predictions))
    

if __name__ == "__main__":
    main()


# moviereviews.tsv
# ar_reviews_100k.tsv