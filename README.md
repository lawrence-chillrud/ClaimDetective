# ClaimDetective

`ClaimDetective` is a python class that allows the user to rank a list of sentences (i.e. potential claims) in order of most check-worthy to least check-worthy.

ClaimDetective was built with a deep-learning model that fine-tunes RoBERTa under-the-hood to identify claims that are worth fact-checking.

## Overview

1. [claim\_detective.py](claim_detective.py) contains all the necessary source code to use the check-worthiness detection models located in [models/](models). 

2. [models](models) is a directory containing the latest trained models. 

3. [sample.py](sample.py) contains a very brief example of loading and using one of the models.

4. [sample\_output.csv](sample_output.csv) shows the saved output after running `sample.py` 

## Models

Each model is located in its own subdirectory. Each model subdirectory contains two files: 

1. `logfile.txt` which contains a log of all the training and testing that model has been through, as well as the architecture of the model.
2. `model.pth` which is a pyTorch checkpoint file containing the model weights in the form of a `state_dict` object.

At the time of writing, the following models are listed: 

* [claimbuster](models/claimbuster) was trained on the [ClaimBuster dataset](https://zenodo.org/record/3609356#.X8q9RxNKhnE) described in [Arslan et. al.](https://arxiv.org/abs/2004.14425) Briefly, the ClaimBuster dataset consists of 23,533 statements extracted from all U.S. general election presidential debates (1960-2016) which were then annotated by human coders.

* [clef19](models/clef19) was trained first on the ClaimBuster dataset described above, and then on the [CLEF-2019 CheckThat! dataset](https://github.com/apepa/clef2019-factchecking-task1#scorers) (CT19-T1 corpus) described in [Atanasova et. al.](https://groups.csail.mit.edu/sls/publications/2019/Mohtarami-CLEF2019.pdf) Briefly, the CT19-T1 corpus contains 23,500 human-annotated sentences from political speeches and debates during the 2016 U.S. presidential election.

* [clef20](models/clef20) was trained solely trained on the [CLEF-2020 CheckThat! dataset](https://github.com/sshaar/clef2020-factchecking-task1#clef2020-checkthat-task-1) (CT20-T1(en) corpus) described in [Barron-Cedeno et. al.](https://arxiv.org/abs/2007.07997) Briefly, the CT20-T1(en) corpus contains 962 human-annotated tweets about the novel coronavirus caused by SARS-CoV-2. 

