# An analysis of Societal Bias in SOTA NLP Transfer Learning

> **NOTE**: This is not a complete recreation of the INLP algorithm, but only a toy example of applying a null projection onto one layer of the encoder. The bias can still be relearnt through the other preceeding layers.

This repository contains code for demonstrating a simplistic application of Nullspace Projection on a NLP Attention-Based Transformer.

The code is largely taken from [pliang279's](https://github.com/pliang279/LM_bias) LM_bias repository and relies heavily on [shauli-ravfogel
's](https://github.com/shauli-ravfogel/nullspace_projection) nullspace projection repository.


## Instructions

#### Prerequisites:
- `Python 3.6.7`
- You will need to clone the [nullspace projection](https://github.com/shauli-ravfogel/nullspace_projection) into this directory before running.


This repo can be accessed primarily via `nullspace_bert_demonstration.ipynb`. This notebook contains cells that install dependencies and generate the appropriate resources. It also runs the code described below.

<br>

#### Discovering Gender Bias Sensitive Tokens

The module `get_bias_sensitive_tokens.py`, uses our predefined gender defining terms to construction a gender bias subspace using PCA. It then takes the highest variance principle component and uses it to discover bias sensitive words in our vocabulary.
These words are printed to the console and corresponding embeddings are saved for future use.

You can run this script with the command:

-  `python get_bias_sensitive_tokens.py`

<br>

#### Discovering a NullSpace Projection

The module `context_nullspace_projection.py` takes the previously discovered embeddings of bias sensitive words and uses their bias direction as a label in a classification task.

It iteratively trains several classifiers on the data, each time generating a projection, `P_i`, that removes the information used by the classifiers weights to linearly separate the embeddings with regards to gender.

You can run this script with the command:

- `python context_nullspace_projection.py`

<br>

## Papers
- [**Lipstick on a Pig: Debiasing Methods Cover up Systematic Gender Biases in Word Embeddings But do not Remove Them**](https://arxiv.org/pdf/1903.03862.pdf) <br>
Hila Gonen and Yoav Goldberg

- [**Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection**](https://aclanthology.org/2020.acl-main.647.pdf) <br>
Shauli Ravfogel, Yanai Elazar, Hila Gonen, Michael Twiton and Yoav Goldberg

- [**Towards Understanding and Mitigating Social Biases in Language Models**](https://arxiv.org/pdf/2106.13219.pdf) <br>
Paul Pu Liang, Chiyu Wu,Louis-Philippe Morency, and Ruslan Salakhutdinov<br>
ICML 2021 <br>

- [**Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings**](https://arxiv.org/pdf/1607.06520.pdf) <br>
Tolga Bolukbasi, Kai-Wei Chang, James Zou, Venkatesh Saligrama and Adam Kalai

- [**Investigating Gender Bias in BERT**](https://arxiv.org/abs/2009.05021) <br>
Rishabh Bhardwaj, Navonil Majumder and Soujanya Poria <br>

- [**BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**](https://arxiv.org/abs/1810.04805) <br>
Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova <br><br>
