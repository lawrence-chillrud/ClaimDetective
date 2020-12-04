# import claim detective model:
from claim_detective import *

# create model:
sherlock = ClaimDetective(path_to_model="./models/claimbuster/model.pth")

# put the sentences / document you would like to check for claims into a list of sentences:
sentences = ["Global warming is causing increasing hurricanes.", "Cats are so much better than dogs."]

# pass the list to the model's inspect method (if you have labels and want to check accuracy, pass the labels. If you don't have labels, labels=None by default):
claims = sherlock.inspect(sents=sentences, labels=[1,0])

# to get the output into a .csv file, pass the output to the report method. file_name sets the output file path name.
sherlock.report(claims, file_name="sample_output.csv")
