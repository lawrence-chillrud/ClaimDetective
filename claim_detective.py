import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from transformers import RobertaModel, RobertaConfig, RobertaTokenizerFast
import pandas as pd
import numpy as np
import time

DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    if iteration == total: 
        print()

class torchDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, length):
        self.encodings = encodings
        self.labels = labels
        self.length = length

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return self.length


class RobertaForClaimDetection(nn.Module):

    def __init__(self, n_classes, unfreeze):
        super(RobertaForClaimDetection, self).__init__()
        self.num_labels = n_classes
        config = RobertaConfig.from_pretrained('roberta-base', output_hidden_states=True)
        self.roberta = RobertaModel.from_pretrained('roberta-base', add_pooling_layer=True, config=config)
        self.drop = nn.Dropout(p=0.1)
        self.out = nn.Linear(self.roberta.config.hidden_size, n_classes)

        for param in self.roberta.base_model.parameters():
            param.requires_grad = unfreeze

    def forward(
        self, 
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        hidden_states = outputs[2]
        roberta_output = torch.div(torch.add(self.roberta.pooler(hidden_states[11]), self.roberta.pooler(hidden_states[12])), 2)
        output = self.drop(roberta_output)
        return self.out(output)

class ClaimDetective():

    def __init__(self, path_to_model):
        # model:
        self.model = RobertaForClaimDetection(n_classes=2, unfreeze=False)
        checkpoint = torch.load(path_to_model)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.model.to(DEVICE)
        
        # tokenizer:
        self.tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

    def inspect(self, sents, labels=None, bs=256):
        tic = time.perf_counter()
        # set up test data:
        test_encodings = self.tokenizer(sents, truncation=True, padding=True)
        test_dataset = torchDataset(encodings=test_encodings, labels=labels, length=len(sents))
        test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False) # batch_size = 1 if n doesn't work
        n = len(test_loader)

        # book-keeping:
        stats = {}
        running_preds = []
        running_softs = []
        running_ones = 0
        running_corrects = 0
        printProgressBar(0, n, prefix='Progress:', suffix='batch 0 / %d' % (n), length=25)
        # test:
        for i, batch in enumerate(test_loader):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            if labels:
                labs = batch['labels'].to(DEVICE)

            with torch.no_grad():
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
             
            softmaxes = F.softmax(logits, dim=1)
            preds = torch.argmax(softmaxes, dim=1)
            running_softs += softmaxes.tolist()
            running_preds += preds.tolist()
            if labels:
                running_corrects += torch.sum(preds == labs.data)
            running_ones += torch.sum(preds)
            printProgressBar(i+1, n, prefix='Progress:', suffix='batch %d / %d' % (i+1, n), length=25)
        print()
        toc = time.perf_counter()
        # summary:
        if labels:
            acc = torch.true_divide(running_corrects, len(labels)).item()
            confusion_mat = confusion_matrix(labels, running_preds)
            cmdf = pd.DataFrame(confusion_mat, columns = ['Pred=0','Pred=1'], index = ['True=0', 'True=1'])
            tp = cmdf.iloc[1, 1]
            precision = tp / (tp + cmdf.iloc[0, 1])
            recall = tp / (tp + cmdf.iloc[1, 0])
            f1 = 2 * (precision * recall)/(precision + recall)

            stats["Accuracy: "] = acc
            stats["Precision: "] = precision
            stats["Recall: "] = recall
            stats["F1 score: "] = f1
            stats["Confusion Matrix: "] = cmdf
            stats["Num. claims detected: "] = "%d out of %d sentences" % (running_ones.item(), len(sents))
            stats["Total time spent classifying %d sentences: " % (len(sents))] = f"{toc - tic:0.4f} seconds"

            print("Accuracy: ", acc)
            print("Precision: ", precision)
            print("Recall: ", recall)
            print("F1 score: ", f1)
            print()
            print("Confusion Matrix: ")
            print(cmdf)
            print()
        print("Num. claims detected: %d of %d sentence(s)" % (running_ones.item(), len(sents)))

        # output:
        ranking_scores = []
        for pair in running_softs:
            ranking_scores.append(pair[1] - pair[0])
        
        if labels:
            ranking = pd.DataFrame.from_records(list(zip(sents, ranking_scores, running_preds, labels)), columns=['Sentence', 'Check-Worthiness Score', 'Prediction', 'Label']).round(5).sort_values(by='Check-Worthiness Score', ascending=False).reset_index(drop=True)
        else:
            ranking = pd.DataFrame.from_records(list(zip(sents, ranking_scores, running_preds)), columns=['Sentence', 'Check-Worthiness Score', 'Prediction']).round(5).sort_values(by='Check-Worthiness Score', ascending=False).reset_index(drop=True)

        return ranking, stats

    def report(self, claims_df, file_name, stats=None):
        claims_df.to_csv(file_name, float_format='%.5f', header=True, index=False)
        print("Saved claims to: ", file_name)

        if stats:
            stats_file_name = file_name.split(sep=".csv")[0] + "_statistics.txt"
            with open(stats_file_name, "w") as fi:
                fi.write("\nStatistics for {}".format(file_name) + "\n\n")
                fi.write("*" * 70 + "\n\n")

                for stat in stats.keys():
                    if stat == "Confusion Matrix: ":
                        line = "\n" + stat + "\n" + str(stats[stat]) + "\n\n"
                    else:
                        line = stat + str(stats[stat]) + "\n"
                    fi.write(line)
            print("Saved stats to: ", stats_file_name)
