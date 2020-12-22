# This takes the big dataset of reviews and returns them without the text review, quartering the size of the file

import json

data = []


for line in open("./Dataset/yelp_dataset/yelp_academic_dataset_review.json", "r"):
    myline = json.loads(line)
    if "text" in myline:
        del myline["text"]
    data.append(myline)

with open("output.json", "w") as outfile:
    json.dump(data, outfile)