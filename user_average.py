import json
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import cosine
import math
import numpy as np
from prompt_toolkit.shortcuts import radiolist_dialog
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def content_based():
    infile = open("output_desc.json", "r")
    data = json.load(infile)
    df = pd.DataFrame.from_dict(data)
    tf = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 1), min_df=0, stop_words="english"
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
    return merge.nlargest(300, "mean")


with open("output_date_short.json") as infile:
    data = json.load(infile)
    df = pd.DataFrame.from_dict(data)
    df = df.drop(columns=["useful", "funny", "cool", "date"])
    business_sort_occurrence = (
        df.groupby(["business_id"]).size().reset_index(name="counts")
    )
    user_occurrence = df.groupby(["user_id"]).size().reset_index(name="user_counts")
    business_sort_occurrence = business_sort_occurrence.nlargest(300, "counts")
    business_sort_occurrence = pd.merge(
        business_sort_occurrence, content_based(), on="business_id"
    )
    merge = pd.merge(df, business_sort_occurrence, on="business_id")
    merge = pd.merge(merge, user_occurrence, on="user_id")
    merge = merge.sort_values(by=["counts"], ascending=False)
    users = merge.nlargest(100, "user_counts").groupby("user_id").mean()
    sample = users.sample(n=10).reset_index()
    mylist = sample["user_id"].tolist()
    mylist = list(map(lambda x: (x, x), mylist))
    chosen_user = radiolist_dialog(
        title="User selection",
        text="Which user would you like to choose",
        values=mylist,
    ).run()
    user_scores = (
        merge.loc[merge["user_id"] == chosen_user]
        .groupby(["business_id"])
        .mean()
        .reset_index()
        .drop(columns=["counts", "user_counts"])
    )


avg_stars = df.drop(columns=["review_id", "user_id"]).groupby(["business_id"]).mean()

business_sort_occurrence = business_sort_occurrence.drop(columns=["counts"])
business_sort_occurrence = pd.merge(
    business_sort_occurrence,
    avg_stars,
    on="business_id",
)

avg_stars = avg_stars.rename(columns={"stars": "avg_stars"})

user_scores = pd.merge(user_scores, avg_stars, on="business_id")


for count, business in enumerate(user_scores["business_id"]):
    cossim = []
    for count1, business1 in enumerate(business_sort_occurrence["business_id"]):
        i1 = (
            df.loc[
                df["business_id"] == business_sort_occurrence.iloc[count]["business_id"]
            ]
            .groupby("user_id")
            .mean()
        )
        i2 = (
            df.loc[
                df["business_id"]
                == business_sort_occurrence.iloc[count1]["business_id"]
            ]
            .groupby("user_id")
            .mean()
        )
        i12 = pd.merge(i1, i2, on="user_id")
        if i12.empty:
            cossim.append(np.nan)
        else:
            cossim.append(1 - cosine(i12["stars_x"], i12["stars_y"]))

    business_sort_occurrence[business] = cossim

# This will remove where all are NaN, because no comparison can be made
for ind in business_sort_occurrence.index:
    nan = (
        math.isnan(business_sort_occurrence[i][ind]) for i in user_scores["business_id"]
    )
    if all(nan):
        business_sort_occurrence = business_sort_occurrence.drop([ind])
scores = []
adj_scores = []
# This generates basic scores, only gives meaningful results if the user has reviewed more than 1 restaurant
for ind in business_sort_occurrence.index:
    top = 0
    bottom = 0
    adj_top = 0
    for i in user_scores.index:
        if (
            math.isnan(
                business_sort_occurrence.at[ind, user_scores.at[i, "business_id"]]
            )
            == False
        ):
            bottom += business_sort_occurrence.at[ind, user_scores.at[i, "business_id"]]
            top += (
                business_sort_occurrence.at[ind, user_scores.at[i, "business_id"]]
                * user_scores.at[i, "stars"]
            )
            adj_top += business_sort_occurrence[user_scores["business_id"][i]][ind] * (
                user_scores["stars"][i] - user_scores["avg_stars"][i]
            )

    scores.append(top / bottom)
    adj_scores.append(business_sort_occurrence["stars"][ind] + (adj_top / bottom))
business_sort_occurrence["weighted_average"] = scores
business_sort_occurrence["adjusted_weighted_average"] = adj_scores
result = business_sort_occurrence[
    ["business_id", "weighted_average", "adjusted_weighted_average"]
]
print(result)
