# import claim detective model:
from claim_detective import *


# create and load the desired model by passing it the name of the 
# pyTorch checkpoint file containing the model:
sherlock = ClaimDetective(path_to_model = "./models/claimbuster/model.pth")


# put the sentences / document you would like to check for claims into a list of sentences:
sentences = ["Global warming is causing increasing hurricanes.", "Cats are so much better than dogs.", "This is the third sentence."]


# pass the above list to the model's inspect() method:
# inspect() takes three arguments: 
# (1) sents = the list of sentences to check for claims
# (2) labels = optional argument that should be a list of the ground truth labels = {0, 1} 
# check-worthy statements are 1's, non-check-worthy are 0's.
# If labels is passed, then inspect() will print summary statistics
# If you don't have labels, labels = None by default. 
# (3) bs = the batch-size to use for classification. bs = 256 by default, 
# and can be increased or decreased as needed (for speed / GPU purposes).
# inspect() will return 2 objects:
# (1) a pandas DataFrame object with 3-4 columns: 
# Sentence, Check-Worthiness Score, Prediction, Label 
# sorted in descending order by Check-Worthiness Score, such that the most
# check-worthy sentences identified as claims appear at the very top of the DataFrame.
# The fourth column, `Label` is only returned if labels is not None.
# (2) a dictionary containing testing statistics if labels were provided.
# if no labels were provided, then the returned stats dictionary will be empty.
labels = [1, 0, 0]
claims, stats = sherlock.inspect(sents=sentences, labels = labels)


# The pandas DataFrame object returned by inspect() 
# can then be saved to a .csv file with the report() method.
# report() takes 3 arguments:
# (1) claims_df = the output of inspect() to be printed.
# (2) file_name = the desired name of the output .csv file. 
# (3) stats = the received stats dictionary from inspect().
# stats is optional and by default = None. If not None, then 
# the testing statistics computed during the inspect() call
# above will be saved to a log file ending in `_statistics.txt`
# located in the same directory as the saved .csv file.
sherlock.report(claims_df=claims, file_name="./example_outputs/small_output.csv", stats=stats)
