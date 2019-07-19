import os
import csv
import random
import shutil

source_dir = "./hasy-data"
train_dir = "./data/train"
valid_dir = "./data/validation"

filepath = "hasy-data-labels.csv"
delimiter=','
quotechar="'"
with open(filepath, 'r', encoding="utf8") as csvfile:
    reader = csv.DictReader(csvfile,
                            delimiter=delimiter,
                            quotechar=quotechar)
    for row in reader:
        row["latex"] = row["latex"].replace('/', '\slash')
        if os.path.exists("%s/%s" % (train_dir, row["latex"])) == False:
            os.makedirs("%s/%s" % (train_dir, row["latex"]))

        if os.path.exists("%s/%s" % (valid_dir, row["latex"])) == False:
            os.makedirs("%s/%s" % (valid_dir, row["latex"]))

        if random.randrange(4) == 0:
            shutil.copyfile("%s" % (row["path"]),
                      "%s/%s/%s" % (valid_dir, row["latex"], os.path.basename(row["path"])))

        else:
            shutil.copyfile("%s" % (row["path"]),
                      "%s/%s/%s" % (train_dir, row["latex"], os.path.basename(row["path"])))
