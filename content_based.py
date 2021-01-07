import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np

with open("output_desc.json", "r") as infile:
    data = json.load(infile)
    df = pd.DataFrame.from_dict(data)
    tf = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 1), min_df=20, stop_words="english"
    )
    tfidf_matrix = tf.fit_transform(df["text"])
    unsparse = pd.DataFrame.sparse.from_spmatrix(tfidf_matrix)
    small = unsparse.iloc[
        df.index[df["user_id"] == "BdRMVROS1MXOHxr-bdZv0g"].tolist(), :
    ]
    reduced_df = df[["business_id"]].reset_index()
    cosine_similarities = linear_kernel(tfidf_matrix, small)
    panda_cos = pd.DataFrame(cosine_similarities).reset_index()
    merge = (
        pd.merge(reduced_df, panda_cos, on="index")
        .drop(columns=["index"])
        .drop(df.index[df["user_id"] == "BdRMVROS1MXOHxr-bdZv0g"].tolist())
    )
    merge["mean"] = merge.mean(axis=1)
    merge = merge[["business_id", "mean"]].groupby("business_id").mean()
    print(merge.sort_values(by=["mean"], ascending=False))