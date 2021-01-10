import json
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import cosine
import math
import numpy as np
from prompt_toolkit.shortcuts import radiolist_dialog
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def content_based(user_id, toremove):
    infile = open("output_desc.json", "r")
    data = json.load(infile)
    df = pd.DataFrame.from_dict(data)
    tf = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 1), min_df=0, stop_words="english"
    )
    tfidf_matrix = tf.fit_transform(df["text"])
    unsparse = pd.DataFrame.sparse.from_spmatrix(tfidf_matrix)
    small = unsparse.iloc[
        df[
            (df["user_id"] == user_id) & (~df["business_id"].isin(toremove))
        ].index.tolist(),
        :,
    ]
    reduced_df = df[["business_id"]].reset_index()
    cosine_similarities = linear_kernel(tfidf_matrix, small)
    panda_cos = pd.DataFrame(cosine_similarities).reset_index()
    merge = pd.merge(reduced_df, panda_cos, on="index").drop(columns=["index"])
    merge["mean"] = merge.mean(axis=1)
    merge = merge[["business_id", "mean"]].groupby("business_id").mean()
    return merge.nlargest(300, "mean")


def collaborative(business_sort_occurrence, user_scores, df):
    shorter_users = df.merge(user_scores, on="business_id", how="inner")
    shorter_business = (
        df.merge(shorter_users["user_id"], on="user_id", how="inner")
        .groupby("business_id")
        .mean()
        .reset_index()
    )
    print(shorter_business)

    for count, business in enumerate(user_scores["business_id"]):
        cossim = []
        for count1, business1 in enumerate(shorter_business["business_id"]):
            i1 = (
                df.loc[df["business_id"] == shorter_business.iloc[count]["business_id"]]
                .groupby("user_id")
                .mean()
            )
            i2 = (
                df.loc[
                    df["business_id"] == shorter_business.iloc[count1]["business_id"]
                ]
                .groupby("user_id")
                .mean()
            )
            i12 = pd.merge(i1, i2, on="user_id", how="inner")
            if i12.empty:
                cossim.append(np.nan)
            else:
                cossim.append(1 - cosine(i12["stars_x"], i12["stars_y"]))

        shorter_business[business] = cossim

    # This will remove where all are NaN, because no comparison can be made
    counter = 0
    for ind in shorter_business.index:
        nan = (math.isnan(shorter_business[i][ind]) for i in user_scores["business_id"])
        if all(nan):
            shorter_business = shorter_business.drop([ind])
            counter += 1
    print("Counter " + str(counter))
    scores = []
    adj_scores = []
    # This generates basic scores, only gives meaningful results if the user has reviewed more than 1 restaurant
    for ind in shorter_business.index:
        top = 0
        bottom = 0
        adj_top = 0
        for i in user_scores.index:
            if (
                math.isnan(shorter_business.at[ind, user_scores.at[i, "business_id"]])
                == False
            ):
                bottom += shorter_business.at[ind, user_scores.at[i, "business_id"]]
                top += (
                    shorter_business.at[ind, user_scores.at[i, "business_id"]]
                    * user_scores.at[i, "stars"]
                )
                adj_top += shorter_business[user_scores["business_id"][i]][ind] * (
                    user_scores["stars"][i] - user_scores["avg_stars"][i]
                )

        scores.append(top / bottom)
        adj_scores.append(shorter_business["stars"][ind] + (adj_top / bottom))
    shorter_business["weighted_average"] = scores
    shorter_business["adjusted_weighted_average"] = adj_scores
    # print(shorter_business)
    result = shorter_business[
        ["business_id", "weighted_average", "adjusted_weighted_average"]
    ]
    return result


with open("output_date_short.json") as infile:
    data = json.load(infile)
df = pd.DataFrame.from_dict(data)
df = df.drop(columns=["useful", "funny", "cool", "date"])


user_occurrence = (
    df.groupby(["user_id", "business_id"])
    .mean()
    .groupby(["user_id"])
    .size()
    .reset_index(name="user_counts")
)

one_users = user_occurrence.loc[user_occurrence["user_counts"] == 1]

df = df[~df["user_id"].isin(one_users["user_id"].tolist())]
print("DF")
print(df)
business_sort_occurrence = df.groupby(["business_id"]).size().reset_index(name="counts")

print(user_occurrence)
# business_sort_occurrence = business_sort_occurrence.nlargest(300, "counts")

merge = pd.merge(df, business_sort_occurrence, on="business_id")
merge = pd.merge(merge, user_occurrence, on="user_id")
merge = merge.sort_values(by=["counts"], ascending=False)
users = merge.groupby("user_id").mean()
users = users.nlargest(10, "user_counts")
sample = users.sample(n=10).reset_index()
mylist = sample["user_id"].tolist()
mylist = list(map(lambda x: (x, x), mylist))
chosen_user = radiolist_dialog(
    title="User selection",
    text="Which user would you like to choose",
    values=mylist,
).run()

toremove = (
    merge.loc[merge["user_id"] == chosen_user]
    .groupby(["business_id"])
    .mean()
    .reset_index()["business_id"]
    .sample(frac=0.5)
    .tolist()
)

full_user_scores = (
    merge.loc[merge["user_id"] == chosen_user]
    .groupby(["business_id"])
    .mean()
    .reset_index()
    .drop(columns=["counts", "user_counts"])
)


indexNames = merge[
    (merge["user_id"] == chosen_user) & (merge["business_id"].isin(toremove))
].index
merge.drop(indexNames, inplace=True)

# merge = pd.merge(
#     merge, content_based(chosen_user, toremove), on="business_id", how="inner"
# )

user_scores = (
    merge.loc[merge["user_id"] == chosen_user]
    .groupby(["business_id"])
    .mean()
    .reset_index()
    .drop(columns=["counts", "user_counts"])
)

print("User scores")
print(user_scores)

avg_stars = df.drop(columns=["review_id", "user_id"]).groupby(["business_id"]).mean()

business_sort_occurrence = business_sort_occurrence.drop(columns=["counts"])
business_sort_occurrence = pd.merge(
    business_sort_occurrence,
    avg_stars,
    on="business_id",
)
# print("Merge:")
# print(merge.groupby(by="business_id").mean())
# print("Business Sort occurence")
# print(business_sort_occurrence)

avg_stars = avg_stars.rename(columns={"stars": "avg_stars"})

user_scores = pd.merge(user_scores, avg_stars, on="business_id")
# merge = merge.drop(columns=["user_counts", "counts", "mean"])
scores = collaborative(business_sort_occurrence, user_scores, df)
# covid_data = []
# for line in open("yelp_academic_dataset_covid_features.json", "r"):
#     myline = json.loads(line)
#     covid_data.append(myline)
# covid_df = pd.DataFrame.from_dict(covid_data).dropna(how="all")
# scores = pd.merge(scores, covid_df, on="business_id")
# collated = scores[scores["Temporary Closed Until"] == "FALSE"]
best = scores.nlargest(20, "adjusted_weighted_average")
infile = open("output_business_names.json")
data = json.load(infile)
business_names = pd.DataFrame.from_dict(data)
collated = pd.merge(best, business_names, on="business_id")
user = []
sugg = []
print(collated)
for index, row in collated.iterrows():
    # if row["Covid Banner"] != "FALSE":
    #     banner = " - " + row["Covid Banner"]
    # else:
    #     banner = ""]
    if row["business_id"] in toremove:
        print(row["business_id"])
