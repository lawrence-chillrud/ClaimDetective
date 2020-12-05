# ClaimDetective

`ClaimDetective` is a python class that allows the user to rank a list of sentences (i.e. potential claims) in order of most check-worthy to least check-worthy, i.e., the priority with which they should be fact-checked.

`ClaimDetective` was built with a deep-learning model that fine-tunes RoBERTa under-the-hood to identify and rank claims that are worth fact-checking.

## Overview

1. [claim\_detective.py](claim_detective.py) contains all the necessary source code to use the check-worthiness detection models located in the `models` directory. 

2. [models](models) is a directory containing the latest trained models. See below for details. 

3. [requirements.txt](requirements.txt) contains the packages and the versions used to write `claim_detective.py`

4. [example\_small.py](example_small.py) contains a very brief example of loading and using one of the models. Read this file before using! Essentially provides all the documentation needed. The output from this file can be found in the `example_outputs` directory, here: [small\_output.csv](example_outputs/small_output.csv). 

5. [example\_big.py](example_big.py) is another example of how to load and use a model in a more realistic setting. Note: to run this you will need more packages than those listed in `requirements.txt` (e.g. `nltk` and `BeautifulSoup`) The output from this file can be found in the `example_outputs` directory, here: [big\_output.csv](example_outputs/big_output.csv)

6. [example\_outputs](example_outputs) contains the output `.csv` files from the two `example.py` files.

## Models

Each model is located in its own subdirectory. Each model subdirectory contains two files: 

1. `logfile.txt` which contains a log of all the training and testing that model has been through, as well as the architecture of the model.
2. `model.pth` which is a pyTorch checkpoint file containing the model weights in the form of a `state_dict` object.

**Because the models are so large, you must download their respective `.zip` files from Google Drive, then unzip each model inside the [models](models) directory.**

At the time of writing, I have made the following models are available on Google Drive: 

* [claimbuster](models/claimbuster.txt) was trained on the [ClaimBuster dataset](https://zenodo.org/record/3609356#.X8q9RxNKhnE) described in [Arslan et. al.](https://arxiv.org/abs/2004.14425) Briefly, the ClaimBuster dataset consists of 23,533 statements extracted from all U.S. general election presidential debates (1960-2016) which were then annotated by human coders.

* [clef19](models/clef19.txt) was trained first on the ClaimBuster dataset described above, and then on the [CLEF-2019 CheckThat! dataset](https://github.com/apepa/clef2019-factchecking-task1#scorers) (CT19-T1 corpus) described in [Atanasova et. al.](https://groups.csail.mit.edu/sls/publications/2019/Mohtarami-CLEF2019.pdf) Briefly, the CT19-T1 corpus contains 23,500 human-annotated sentences from political speeches and debates during the 2016 U.S. presidential election.

* [clef20](models/clef20.txt) was trained solely trained on the [CLEF-2020 CheckThat! dataset](https://github.com/sshaar/clef2020-factchecking-task1#clef2020-checkthat-task-1) (CT20-T1(en) corpus) described in [Barron-Cedeno et. al.](https://arxiv.org/abs/2007.07997) Briefly, the CT20-T1(en) corpus contains 962 human-annotated tweets about the novel coronavirus caused by SARS-CoV-2. 

Note that the very first time running a model will take a few minutes to load and run everything properly. After that first go, using the model to identify claims is very fast.
