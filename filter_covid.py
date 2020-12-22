# This takes the list of companies that have information about their situation due to covid and returns those that are italian restaurants

import json
from tqdm import tqdm

data = []
business = open("output_business.json", "r")
business_data = json.load(business)

with open("covid_usable.json") as infile:
    data1 = json.load(infile)
    for myline in tqdm(data1):
        if myline["business_id"] in business_data:
            for key in myline:
                if myline[key] == "FALSE":
                    myline[key] = False
                if myline[key] == "TRUE":
                    myline[key] = True
            data.append(myline)

with open("output_covid.json", "w") as outfile:
    json.dump(data, outfile, indent=2)