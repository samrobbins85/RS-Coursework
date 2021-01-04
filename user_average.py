import json
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import cosine
import math
import numpy as np
from prompt_toolkit.shortcuts import radiolist_dialog


with open("output_date_short.json") as infile:
    data = json.load(infile)
    df = pd.DataFrame.from_dict(data)
    df = df.drop(columns=["useful", "funny", "cool", "date"])
    business_sort_occurrence = (
        df.groupby(["business_id"]).size().reset_index(name="counts")
    )
    user_occurrence = df.groupby(["user_id"]).size().reset_index(name="user_counts")
    business_sort_occurrence = business_sort_occurrence.nlargest(50, "counts")
    merge = pd.merge(df, business_sort_occurrence, on="business_id")
    merge = pd.merge(merge, user_occurrence, on="user_id")
    merge = merge.sort_values(by=["counts"], ascending=False)
    # print(merge)
    # print(merge)
    users = merge.nlargest(500, "user_counts").groupby("user_id").mean()
    # This will give a random sample of 10 of the 500 users who review italian restaurants most
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
    print(user_scores)


business_sort_occurrence = business_sort_occurrence.drop(columns=["counts"])
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
print(business_sort_occurrence)
