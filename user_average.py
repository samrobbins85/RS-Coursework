import json
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import cosine
import math
import numpy as np

with open("output_date_short.json") as infile:
    data = json.load(infile)
    df = pd.DataFrame.from_dict(data)
    df = df.drop(columns=["useful", "funny", "cool", "date"])
    business_sort_occurrence = (
        df.groupby(["business_id"]).size().reset_index(name="counts")
    )
    business_sort_occurrence = business_sort_occurrence.nlargest(50, "counts")
    merge = pd.merge(df, business_sort_occurrence, on="business_id")
    merge = merge.sort_values(by=["counts"], ascending=False)
    # print(merge)
    print(business_sort_occurrence.iloc[0]["business_id"])
    # i1 = (
    #     df.loc[df["business_id"] == business_sort_occurrence.iloc[0]["business_id"]]
    #     .groupby("user_id")
    #     .mean()
    # )
    # # i1 = i1.groupby("user_id").mean()
    # i2 = (
    #     df.loc[df["business_id"] == business_sort_occurrence.iloc[3]["business_id"]]
    #     .groupby("user_id")
    #     .mean()
    # )
    # i12 = pd.merge(i1, i2, on="user_id")
    # # This is the dot product
    # dot = i12.stars_x.dot(i12.stars_y)
    # # This is the sqrt of squares
    # sqrtsqi1 = math.sqrt(i12["stars_x"].pow(2).sum())
    # sqrtsqi2 = math.sqrt(i12["stars_y"].pow(2).sum())
    # # This is the cosine similarity
    # print(dot / (sqrtsqi1 * sqrtsqi2))
    # print(1 - cosine(i12["stars_x"], i12["stars_y"]))
    business_sort_occurrence = business_sort_occurrence.drop(columns=["counts"])
    for count, business in enumerate(business_sort_occurrence["business_id"]):
        cossim = []
        for count1, business1 in enumerate(business_sort_occurrence["business_id"]):
            i1 = (
                df.loc[
                    df["business_id"]
                    == business_sort_occurrence.iloc[count]["business_id"]
                ]
                .groupby("user_id")
                .mean()
            )
            # i1 = i1.groupby("user_id").mean()
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

    # users = []
    # all_user = panda.user_id.unique()
    # for user in all_user[:100]:
    #     subset = panda.loc[panda["user_id"] == user]
    #     users.append({user: subset["stars"].mean()})
    # print(users[:50])