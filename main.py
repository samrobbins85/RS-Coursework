import json

data = []
for line in open("./Dataset/yelp_dataset/yelp_academic_dataset_business.json", "r"):
    data.append(json.loads(line))

print(data[0])