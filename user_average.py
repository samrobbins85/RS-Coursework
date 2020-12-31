import json
import pandas as pd
from tqdm import tqdm

with open("output_date_short.json") as infile:
    data = json.load(infile)
    df = pd.DataFrame.from_dict(data)
    business_sort_occurrence = (
        df.groupby(["business_id"]).size().reset_index(name="counts")
    )
    business_sort_occurrence = business_sort_occurrence.nlargest(50, "counts")
    merge = pd.merge(df, business_sort_occurrence, on="business_id")
    merge = merge.sort_values(by=["counts"], ascending=False)
    print(merge)
    # users = []
    # all_user = panda.user_id.unique()
    # for user in all_user[:100]:
    #     subset = panda.loc[panda["user_id"] == user]
    #     users.append({user: subset["stars"].mean()})
    # print(users[:50])