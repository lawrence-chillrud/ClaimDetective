# import claim detective model:
from claim_detective import *


# create and load the desired model by passing it the name of the 
# pyTorch checkpoint file containing the model:
sherlock = ClaimDetective(path_to_model = "./models/claimbuster/model.pth")


# put the sentences / document you would like to check for claims into a list of sentences:
sentences = ["Global warming is causing increasing hurricanes.", "Cats are so much better than dogs.", "This is the third sentence."]


# pass the above list to the model's inspect() method:
# inspect() will return a pandas DataFrame object with 3 columns: 
# Sentence, Check-Worthiness Score, Prediction
# inspect() takes two arguments: 
# (1) sents = the list to check for claims
# (2) labels = optional argument that should contain a list of the ground truth labels = {0, 1} 
# check-worthy statements are 1's, non-check-worthy are 0's.
# If labels is passed, then inspect() will print accuracy statistics & a confusion matrix
# If you don't have labels, labels = None by default. 
labels = [1, 0, 0]
claims = sherlock.inspect(sents=sentences, labels = labels)


# The pandas DataFrame object returned by inspect() 
# can then be saved to a .csv file with the report() method.
# report() takes two arguments:
# (1) claims_df = the output of inspect() to be printed.
# (2) file_name = the desired name of the output .csv file. 
sherlock.report(claims_df=claims, file_name="./example_outputs/small_output.csv")
