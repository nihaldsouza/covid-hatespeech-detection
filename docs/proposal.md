# Introduction: 
Following the outbreak of COVID-19, and the identification of its origins in China, there has been a surge in anti-Asian hate crimes ranging from microaggressions to physical and verbal assault (Montemurro, 2020). He et al (2021) address this problem by creating a anti-Asian hate and counterspeech dataset containing over 206 million tweets. This group was able to train classifiers with BERT embeddings to identify between hate speech, counter speech and neutral with an F1 score of 0.832. We hope to expand on the work by He et al (2021) by using transfer learning. We hope to train our model on hate speech that is not specific to COVID-19 and anti-asian hate before using the dataset created by He et al. We also hope to analyze the results of our model and visualize where these tweets are coming from geographically in order to assess whether there are any area that have disproportionately high anti-Asian tweets.
#### Research Question
Is there something unique about the COVID-19 anti-Asian hatespeech such that transfer learning from more general hate speech hinders the performance of the model from He et al?

#### Hypothesis
We hypothesis that using transfer learning from datasets of  non specific to anti-asian/covid hatespeech and counter speech will augment the f1 scores observed by He et al.

# Motivation and Contribution

Since the start of the COVID-19 pandemic, there has been a widespread increase in the amount of hate-speech being propogated online against the Asian community. This type of toxic language can trigger harmful real-world events, and have long time consequences. This project seeks to understand if general form of hate-speech has patterns that could be used to identify COVID-19 hate speech. We would also explore the geographical extent across certain key events and time periods.

# Data:

We have three different datasets containing tweets. All of these datasets were found online, so we will not be scraping and annotating data ourselves. 

Two of our datasets are from before the COVID-19 pandemic. 
These two datasets are the [Hate Speech and Offensive Language dataset](https://github.com/t-davidson/hate-speech-and-offensive-language) and
the [Hate and Abusive Speech on Twitter dataset](https://github.com/ENCASEH2020/hatespeech-twitter)

The Hate Speech and Offensive Language dataset contains 24,802 annotated tweets collected in 2017. These tweets were annotated as either hate, offensive, or neither. 
Overall, there were 19190 tweets labeled as offensive language, 4163 labeled as niether hate or offensive, and 1430 labeled as hate speech. Each tweet was annotated by 3 annotater on average.

The Hate and Abusive Speech on Twitter dataset contains 100,000 annotated tweets collected in 2018. These tweets were annotated as either normal, abusive, spam, or hateful. 
Overall, there were 53851 tweets labeled as normal, 27150 tweets labeled as abusive, 14030 labeled as spam, and 4965 tweets labeled as hateful. 
Each tweet was annotated by 3 annotaters on average. 

Both of these datasets have a class imbalance. In the first dataset, majority of the tweets are labeled as offensive. In the second dataset, majority of the tweets are labeled as normal. 
In both of these datasets, the number of tweets labeled as hate is very low compared to the other classes. 

The last dataset we are using is the COVID-HATE dataset, which is mentioned in the project description. This dataset has over 206 million tweets from between January 15, 2020, and March 26, 2021. 
3355 of these tweets were hand-annotated as hate speech, counterspeech or neutral. 
Of these hand-annotated tweets, 1344 are labeled as neutral, 517 are labeled as counter-hate, and 429 are labeled as hatespeech. 
From the tweets labeled by BERT, there are 203857160 labeled as neutral, 1337116 labeled as counter-hate, and 1154289 labeled as hatespeech. We will additionally be using 21,414 of the BERT labeled tweets from this dataset. 

All of these datasets use different labels. We will be changing the labels so that all three datasets are annotated the same way. 
This will include changing the 'hate', 'hateful' and 'hatespeech' labels to be the same and changing the 'neither', 'neutral' and 'normal' labels to the same. 
We will have to remove the 'spam' tweets from the Hate and Abusive Speech on Twitter dataset since no other dataset has a similar label. 

We created Wordclouds for each of the datasets to see what the most common words contained in the tweets were. This can be found in the ```corpus_stats.ipynb``` file. 


# Methods and Engineering:

## Data Preprocessing
The dataset from He et al contains the tweet ID and the label that was given by BERT. We used the Twitter API to locate these tweets using the ID. In order to see the affect of using a general dataset and then fine tuning on a the task specific dataset, we are not using the full 2 million tweets from the COVID anti-Asian dataset. There are a couple reasons for this:
    1. The twitter API only allows 900 API calls very 15 minutes so it would take a lot of time to get all 2 million tweets
    2. It would also be very computationally draining to train with such a huge dataset
    3. We fear that it may drown out the generic dataset which would not help us address our research question.
    
The code used to perform this subsetting of the data can be found [here](SubsettingData.ipynb)

## Engineering

- We have implemented the BERT model from [Racism is a Virus: Anti-Asian Hate and
Counterspeech in Social Media during the
COVID-19 Crisis](https://arxiv.org/pdf/2005.12423.pdf) by Bing He et al. 
- Each tweet is embedded using the [BERT base uncased text embedding model](https://arxiv.org/pdf/1810.04805.pdf) resulting in an embedding representation of each tweet. Next, the embedding is used as an input to a neural network classifier with one feed-forward layer. Finally, the BERT classification model is fine-tuned as fine tuning provides superior classification performance.
- For the training process, the authors employ one-layer feed-forward neural network, similar to the BERT model. They train this model using the linguistic and hashtag features of the tweet. For hyper parameter optimization, they employ grid search to find the optimal hyper-parameters as: 
  - Batch size: 8
  - Epochs: 3
  - Learning rate: 1e-5
  
  that led to the best classification results. Also, five fold cross validation was conducted on the hand-annotated dataset.
- For metrics, precision, recall and F1 score were employed. Comparing the performance of BERT embedded, linguistic and hashtag features, the authors concluded that the BERT embedded features had the best F1 score overall.
- Also, in categorized tweets, the BERT model had high precision and recall in categorizing the tweets. The advantage being that the BERT classifier prevents the model from being overly reliant on the presence of specific hashtags and keywords, whose popularity may change over time.

Current work is in the following [file](bert_covid_hate.ipynb)


For this project, we are using both Jupyterlab on our personal computers and Google Colab. We will be employing text classification with BERT and using the PyTorch Framework to do transfer learning.

# Previous work
There have been several previous projects looking at hate speech, some specifically using data collected since the beginning of the COVID-19 pandemic, some looking at transfer learning and others on just general hate speech classification. He et al (2021) create a dataset specifically for classification of hate and counterspeech for anti-Asian hate and use BERT embeddings to create a classification model. Vishwamitra et al (2020) also created a COVID-19 dataset that consisted of tweets targeting older people and supplemented with a dataset targeting the Asian community. Their goal was to discover specific keywords used in hate speech against these groups. In order to prepare for this project, we wanted to look at different work using transfer learning. Benítez-Andrades et al (2022) used BERT embeddings along with attention mechanisms. This group first trained a model on an English dataset containing more general hate speech and then use transfer learning with a second Spanish dataset to achieve better classification results. Another study by Boy et al (2021) uses transfer learning for sentiment tasks by taking models with parameters trained for emoji-based tasks. Wei et al (2021) uses a pretrained BERT model and used transfer learning by freezing all the layers of the model and attached a few neural layers of their own and train that part. This array of previous work makes us confidant that we will be able to achieve good results with transferring what the model learned in the generic hate speech dataset to the COVID specific one.

# Evaluation

Since the model task is binary classification, we will be using macro-F1 score as our metric. We would also employ a Choropleth for visualizing the extent of hate-speech.

# Challenges
A challenge we ran into when trying to get the data was that the actual tweets were not stored but rather only the tweet IDs. This was unfortunate because it meant we had to locate the tweets ourselves and this was a long process since the Twitter API only allows 900 API calls every 15 minutes. Another thing we noted from the dataset was that since it was the tweet IDs and not the actual tweets, several of the tweets had been deleted since they had collected them.


# References
Benítez-Andrades, J. A., González-Jiménez, Á., López-Brea, Á., Aveleira-Mata, J., Alija-Pérez, J. M., & García-Ordás, M. T. (2022). Detecting racism and xenophobia using deep learning models on Twitter data: CNN, LSTM and BERT. PeerJ Computer Science, 8, e906.

Boy, Susann, Dana Ruiter, and Dietrich Klakow. "Emoji-based transfer learning for sentiment tasks." arXiv preprint arXiv:2102.06423 (2021).

He, B., Ziems, C., Soni, S., Ramakrishnan, N., Yang, D., & Kumar, S. (2021, November). Racism is a virus: anti-asian hate and counterspeech in social media during the COVID-19 crisis. In Proceedings of the 2021 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining (pp. 90-94).

Jafari, Amir Reza, et al. "Transfer Learning for Multi-lingual Tasks--a Survey." arXiv preprint arXiv:2110.02052 (2021).

N. Montemurro, “The emotional impact of covid-19:
from medical staff to common people,” Brain, behavior,
and immunity, 2020.

Nozza, Debora. "Exposing the limits of zero-shot cross-lingual hate speech detection." Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers). 2021.

Vishwamitra, N., Hu, R. R., Luo, F., Cheng, L., Costello, M., & Yang, Y. (2020, December). On analyzing covid-19-related hate speech using bert attention. In 2020 19th IEEE International Conference on Machine Learning and Applications (ICMLA) (pp. 669-676). IEEE.

Wei, Bencheng, et al. "Offensive Language and Hate Speech Detection with Deep Learning and Transfer Learning." arXiv preprint arXiv:2108.03305 (2021).
