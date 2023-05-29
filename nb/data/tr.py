# help me write a function that translates a json file to csv
# I want to be able to specify the json file and the csv file

import json
import csv
import sys

def translate(json_file, csv_file):
    # open the json file
    json_data = open(json_file)
    data = json.load(json_data)
    json_data.close()

    # open the csv file
    csv_data = open(csv_file, 'w')
    csvwriter = csv.writer(csv_data)

    # write the header
    csvwriter.writerow(data[0].keys())

    # write the data
    for row in data:
        csvwriter.writerow(row.values())

    csv_data.close()

if __name__ == '__main__':
    translate(sys.argv[1], sys.argv[2])

# Path: data/tr.py