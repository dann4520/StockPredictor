import csv

with open("/home/daniel/Downloads/AAPL.csv", "rb") as csvfile:
	spamreader = csv.reader(csvfile)
	for row in spamreader:
		print(row)