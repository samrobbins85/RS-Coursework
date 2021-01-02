import json
import pandas as pd
from tqdm import tqdm

with open("output_date_short.json") as infile:
    data = json.load(infile)
    df = pd.DataFrame.from_dict(data)
    df = df.drop(columns=["useful", "funny", "cool", "date"])
    business_sort_occurrence = (
        df.groupby(["business_id"]).size().reset_index(name="counts")
    )
    business_sort_occurrence = business_sort_occurrence.nlargest(5, "counts")
    merge = pd.merge(df, business_sort_occurrence, on="business_id")
    merge = merge.sort_values(by=["counts"], ascending=False)
    # print(merge)
    print(business_sort_occurrence.iloc[0]["business_id"])
    i1 = (
        df.loc[df["business_id"] == business_sort_occurrence.iloc[0]["business_id"]]
        .groupby("user_id")
        .mean()
    )
    # i1 = i1.groupby("user_id").mean()
    i2 = (
        df.loc[df["business_id"] == business_sort_occurrence.iloc[3]["business_id"]]
        .groupby("user_id")
        .mean()
    )
    i12 = pd.merge(i1, i2, on="user_id")
    print(i12)
    # users = []
    # all_user = panda.user_id.unique()
    # for user in all_user[:100]:
    #     subset = panda.loc[panda["user_id"] == user]
    #     users.append({user: subset["stars"].mean()})
    # print(users[:50])