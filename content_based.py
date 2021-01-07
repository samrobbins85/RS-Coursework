import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

with open("output_desc.json", "r") as infile:
    data = json.load(infile)
    df = pd.DataFrame.from_dict(data)
    tf = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 3), min_df=0, stop_words="english"
    )
    tfidf_matrix = tf.fit_transform(df["text"])
    print(tfidf_matrix)