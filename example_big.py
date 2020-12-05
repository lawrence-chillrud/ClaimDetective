from claim_detective import *
from nltk.tokenize import sent_tokenize
from bs4 import BeautifulSoup
import requests

def get_webpage(url="https://www.bbc.com/news/science-environment-51742646"):
    r = requests.get(url)
    soup = BeautifulSoup(r.content, features="html.parser")
    paras = soup.find_all('p')
    sents = []
    for p in paras:
        if p:
            sents += sent_tokenize(p.get_text())

    return sents

if __name__ == "__main__":
    sherlock = ClaimDetective("./models/claimbuster/model.pth")
    print("Loaded model! Now attempting to scrape the webpage...")
    sents = get_webpage()
    print("Got the webpage! Now attempting to find claims...")
    claims = sherlock.inspect(sents)
    print("Saving claims!")
    sherlock.report(claims, file_name="./example_outputs/big_output_claimbuster.csv")

