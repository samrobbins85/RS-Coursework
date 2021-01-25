import json
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import cosine
import math
import numpy as np
from prompt_toolkit.shortcuts import radiolist_dialog
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics import confusion_matrix


def content_based(user_id):
    infile = open("output_desc.json", "r")
    data = json.load(infile)
    data = pd.DataFrame.from_dict(data)
    tf = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 1), min_df=0, stop_words="english"
    )
    tfidf_matrix = tf.fit_transform(data["text"])
    unsparse = pd.DataFrame.sparse.from_spmatrix(tfidf_matrix)
    small = unsparse.iloc[
        data[data["user_id"] == user_id].index.tolist(),
        :,
    ]
    reduced_df = data[["business_id"]].reset_index()
    cosine_similarities = linear_kernel(tfidf_matrix, small)
    panda_cos = pd.DataFrame(cosine_similarities).reset_index()
    merge = pd.merge(reduced_df, panda_cos, on="index").drop(columns=["index"])
    merge["mean"] = merge.mean(axis=1)
    merge = merge[["business_id", "mean"]].groupby("business_id").mean()
    return merge


def collaborative(business_sort_occurrence, user_scores, df):
    shorter_users = df.merge(user_scores, on="business_id", how="inner")
    shorter_business = (
        df.merge(shorter_users["user_id"], on="user_id", how="inner")
        .groupby("business_id")
        .mean()
        .reset_index()
    )

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
    # scores = []
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

        # scores.append(top / bottom)
        adj_scores.append(shorter_business["stars"][ind] + (adj_top / bottom))
    # shorter_business["weighted_average"] = scores
    shorter_business["adjusted_weighted_average"] = adj_scores
    result = shorter_business[["business_id", "adjusted_weighted_average"]]
    return result


with open("output_date_short.json") as infile:
    data = json.load(infile)
df = pd.DataFrame.from_dict(data)
df = df.drop(columns=["useful", "funny", "cool", "date"])
# How many reviews a user has left
user_occurrence = (
    df.groupby(["user_id", "business_id"])
    .mean()
    .groupby(["user_id"])
    .size()
    .reset_index(name="user_counts")
)
# Users that just leave one review
one_users = user_occurrence.loc[user_occurrence["user_counts"] == 1]
# Remove from dataframe
df = df[~df["user_id"].isin(one_users["user_id"].tolist())]
# Group by the business, and get a count of how many reviews they have
business_sort_occurrence = df.groupby(["business_id"]).size().reset_index(name="counts")
# Add this counts column to merge
merge = pd.merge(df, business_sort_occurrence, on="business_id")
# Also get the user counts
merge = pd.merge(merge, user_occurrence, on="user_id")
# Get the average score each user leaves
users = merge.groupby("user_id").mean()
# Choose the n users that have reviewed the most restaurants
users = users.nlargest(10, "user_counts")
# Sample n1 from that selection
sample = users.sample(n=10).reset_index()
# Turn this dataframe to a list to display to the user
mylist = sample["user_id"].tolist()
mylist = list(map(lambda x: (x, x), mylist))
chosen_user = radiolist_dialog(
    title="User selection",
    text="Which user would you like to choose",
    values=mylist,
).run()

with open("output_date_short.json") as infile:
    data = json.load(infile)
df = pd.DataFrame.from_dict(data)
df = df.drop(columns=["useful", "funny", "cool", "date"])
# How many reviews a user has left
user_occurrence = (
    df.groupby(["user_id", "business_id"])
    .mean()
    .groupby(["user_id"])
    .size()
    .reset_index(name="user_counts")
)
# Users that just leave one review
one_users = user_occurrence.loc[user_occurrence["user_counts"] == 1]
# Remove from dataframe
df = df[~df["user_id"].isin(one_users["user_id"].tolist())]
# Group by the business, and get a count of how many reviews they have
business_sort_occurrence = df.groupby(["business_id"]).size().reset_index(name="counts")
# Add this counts column to merge
merge = pd.merge(df, business_sort_occurrence, on="business_id")
# Also get the user counts
merge = pd.merge(merge, user_occurrence, on="user_id")
# Get the average score each user leaves
users = merge.groupby("user_id").mean()
# Choose the n users that have reviewed the most restaurants
users = users.nlargest(10, "user_counts")
eval_user_scores = (
    merge.loc[merge["user_id"] == chosen_user]
    .groupby(["business_id"])
    .mean()
    .reset_index()
    .drop(columns=["counts", "user_counts"])
)
# For the specified user, now get their ratings of restaurants
user_scores = (
    merge.loc[merge["user_id"] == chosen_user]
    .groupby(["business_id"])
    .mean()
    .reset_index()
    .drop(columns=["counts", "user_counts"])
)
# Average rating of each restaurant
avg_stars = df.drop(columns=["review_id", "user_id"]).groupby(["business_id"]).mean()
# Now make business sort occurence feature those stars
business_sort_occurrence = business_sort_occurrence.drop(columns=["counts"])
business_sort_occurrence = pd.merge(
    business_sort_occurrence,
    avg_stars,
    on="business_id",
)
# Change the name so it merges nicely
avg_stars = avg_stars.rename(columns={"stars": "avg_stars"})
# For the places the user has rated, also have the average rating of that place
user_scores = pd.merge(user_scores, avg_stars, on="business_id")
scores = collaborative(business_sort_occurrence, user_scores, df)
content = content_based(chosen_user)
scores = pd.merge(scores, content, on="business_id", how="inner")
scores["adjusted_weighted_average"] = (
    scores["adjusted_weighted_average"] - scores["adjusted_weighted_average"].min()
) / (
    scores["adjusted_weighted_average"].max()
    - scores["adjusted_weighted_average"].min()
)
scores["mean"] = (scores["mean"] - scores["mean"].min()) / (
    scores["mean"].max() - scores["mean"].min()
)
scores["merge"] = scores["mean"] + scores["adjusted_weighted_average"]
covid_data = []
for line in open("yelp_academic_dataset_covid_features.json", "r"):
    myline = json.loads(line)
    covid_data.append(myline)
covid_df = pd.DataFrame.from_dict(covid_data).dropna(how="all")
scores = pd.merge(scores, covid_df, on="business_id")
collated = scores[scores["Temporary Closed Until"] == "FALSE"]
best = collated.nlargest(5, "merge")
infile = open("output_business_names.json")
data = json.load(infile)
business_names = pd.DataFrame.from_dict(data)
collated = pd.merge(best, business_names, on="business_id")
for index, row in collated.iterrows():
    print(row["name"] + " " + str(int(round(row["merge"] * 50, 0))) + "%")
