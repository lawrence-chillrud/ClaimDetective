# import claim detective model:
from claim_detective import *
import pandas as pd
import os
import glob

def load_ClaimBuster_data(train=False):
    # groundtruth.csv and crowdsourced.csv
    if train:
        df = pd.read_csv("../ClaimDetection/ClaimBuster_Datasets/datasets/crowdsourced.csv")
    else:
        df = pd.read_csv("../ClaimDetection/ClaimBuster_Datasets/datasets/groundtruth.csv")
    #df = df1.append(df2, ignore_index=True) # for both...
    texts = df['Text']
    labels = df['Verdict'].copy()
    labels[labels != 1] = 0
    ids = df['Sentence_id']
    return texts.tolist(), labels.tolist(), ids.tolist()

def load_clef20_data(data_file="../ClaimDetection/clef2020-factchecking-task1/test-input/test-gold.tsv"):
    df = pd.read_csv(data_file, sep='\t')
    texts = df['tweet_text'].tolist()
    labels = df['check_worthiness'].tolist()
    #topic_ids = df['topic_id'].tolist()
    tweet_ids = df['tweet_id'].tolist()
    return texts, labels, tweet_ids

# to load data from CLEF
def load_clef19_data(datadir='../ClaimDetection/clef2019-factchecking-task1/data/test_annotated/*.tsv', as_docs=False):
    files = glob.glob(datadir)
    texts = list()
    labels = list()
    ids = list()
    for f in files:
        df = pd.read_csv(f, sep='\t', names=['no.', 'speaker', 'statement', 'label'])
        if as_docs:
            texts.append(df['statement'])
            labels.append(df['label']) # for 1 class model
            #lbls = [((0, 1)) if lbl else ((1, 0)) for lbl in df['label']] # for 2 class model
            #labels.append(lbls) # for 2 class model
            ids.append(df['no.'])
        else:
            texts += df['statement'].tolist()
            labels += df['label'].tolist() # for 1 class model
            #lbls = [((0, 1)) if lbl else ((1, 0)) for lbl in df['label']] # for 2 class model
            #labels += lbls # for 2 class model
            ids += df['no.'].tolist()
    return texts, labels, ids

if __name__ == "__main__":
    
    if not os.path.isdir('./incorrect_preds/'):
        os.makedirs('./incorrect_preds/')
    
    models = {'claimbuster': load_ClaimBuster_data, 'clef19' : load_clef19_data, 'clef20': load_clef20_data}

    for m in models.keys():
        # import model:
        print("Loading {} model...".format(m))
        sherlock = ClaimDetective(path_to_model = "./models/{}/model.pth".format(m))

        # import test data:
        print("Loading test data...")
        test_x, test_y, test_ids = models[m]()
        
        # inspect sentences for claims:
        print("Inspecting claims...")
        claims, stats = sherlock.inspect(sents=test_x, labels=test_y)
        
        # only retrieve those sentences that were misclassified:
        misclass_claims = claims[claims.Prediction != claims.Label].sort_values(by='Check-Worthiness Score', ascending=False, key=abs).reset_index(drop=True)
        print("Saving those claims that were misclassified...")
        sherlock.report(misclass_claims, file_name="./incorrect_preds/{}.csv".format(m), stats=stats)

