#!/usr/bin/env python
# coding: utf-8

# #idea
# 
# - Word embeding
# - convert text to other means
# - link features list of nmaek so each to trace what each column means
# - do i see corellation between keyword coloumn and text?
# 
# 

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import gc

import tqdm.autonotebook

import textfeture as txtf

from bs4 import BeautifulSoup


import regex as re
import string

import pickle
import os


from flair.data import Sentence




# In[6]:


create_text_emb = False
#create_text_label = True couldnt make it create valid info
#create_loc_label = True

run_create_feature=True

# In[7]:


os.system('pwd')
colab = False
if os.path.exists('/content/'):
    os.chdir("/content/")
    os.makedirs('data', exist_ok=True)

    colab=True
    os.system("pip install flair")

else:
    os.chdir("/Users/adiel/work/twitter-nlp")

os.system('pwd')


# In[8]:


data_filename = 'data/train.csv'
all_csv = pd.read_csv(data_filename)


# In[9]:


# all_csv.isna().value_counts()


# In[10]:


x=all_csv.target.value_counts()
y=all_csv.target.unique()
print(y)
plt.bar(y,x)


# In[11]:


all_csv.head()


# In[12]:


all_csv.nunique()


# In[13]:


all_csv.groupby('keyword').count()


# In[14]:


all_csv.replace(np.NaN, '', inplace=True)

if (run_create_feature):

    # all_csv = all_csv.sample(n=4, weights='target')

    def clean_text(tstr):
        tstr = tstr.strip()
        tstr = BeautifulSoup(tstr, 'html.parser').get_text()
        tstr = re.sub(r'https?:\/\/[\w\.\/]+', 'URL', tstr)
        tstr = re.sub("@\w+","USER",tstr)

        # Words with punctuations and special characters
        for p in string.punctuation:
            tstr = tstr.replace(p, f' {p} ')

        tstr = tstr.replace("=>"," at ")
        tstr = re.sub("\s+"," ",tstr)
        return tstr

    tstr = "#RockyFire Update => California Hwy. 20 closed in bot #ASD rt notexplained: the only known image of infamous hijacker d.b. cooper. http://t.co/jlzk2hdetg asdasd http://t.co/jlzk2hdetg "

    print("org:", string.punctuation)
    print("op1:",clean_text(tstr))
    print("op2",txtf.clean_text_before_save(tstr))



    # In[15]:


    for idx, row in all_csv.iterrows():
        all_csv.loc[idx, 'text'] = "keyword " + row['keyword'] +" location " + row['keyword'] + " text " + row['text']

    for idx, row in all_csv.iterrows():
        all_csv.loc[idx,'text_clean2'] = clean_text(row['text'])
        all_csv.loc[idx,'text_clean'] = txtf.clean_text_before_save(row['text'])
        # result = {}
        debug=0
        html_orj = row['text']
        html_data = txtf.get_text_from_html(html_orj, debug=debug)
        result = txtf.get_count_from_html(html_data, result=html_data, debug=debug)
        result = txtf.get_nlp_count_from_text(html_data['clean_text'], result=result, debug=debug)
        result = txtf.punctuation_count(html_data['text'], result=result, debug=debug)
        # result = txtf.get_document_embeding(html_data['clean_text'], result=result, debug=debug)
        #print(f"idx:{idx} == {result}")
        #txtf.single_doc_tfidf(text=html_orj, text_vec=text_vec, debug=1)
        all_csv.loc[idx,"text_result"] = [result]


    #get the text column
    all_text = all_csv['text_clean'].tolist()
    text_vec = txtf.count_vector_most(all_text, debug=0)
    tfidf_vec={}
    for idx, row in all_csv.iterrows():
        result = txtf.single_doc_tfidf(text=row['text_clean'], text_vec=text_vec)
        tfidf_vec[idx] = result['tf_idf_vector']
    all_csv['tfidf_vec'] = tfidf_vec

    # In[16]:

    os.makedirs('tmp', exist_ok=True)
    all_csv.to_pickle("tmp/all_csv.pkl")

else:
    print("create features was not used")




# In[17]:


#Load the saves aside data
def load_pkl(file_name):
  path = os.path.join(file_name)
  with open(path, 'rb') as data:
    output = pickle.load(data)
  return output

all_csv = load_pkl("tmp/all_csv.pkl")



# prepare the X and Y before split into test and valid
x = all_csv.drop(columns=['target','id', 'text_clean', 'text_clean2', 'location', 'keyword' ])

print(x.head())
print(x.describe())
print(x.info())


y = all_csv['target']


# In[20]:

from sklearn.model_selection import train_test_split
if (len(x) > 100):
    stratify=y
else:
    stratify = None

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1, stratify=stratify)



# text_emb_fname = 'tmp/X_train_test_clean.pkl'
# if (create_text_emb):
#     from flair.embeddings import TransformerDocumentEmbeddings
#     # init embedding
#     X_train = load_pkl("tmp/X_train.pkl")
#
#     # create a sentence
#     tmp_data = {}
#     for idx, row in X_train.iterrows():
#         gc.collect()
#         tstr = row['text_clean']
#         # tstr = tstr.lower()
#         print("S ",idx," ",tstr)
#         embedding = TransformerDocumentEmbeddings('bert-base-cased')
#         sentence = Sentence(tstr)
#
#         # embed the sentence
#         embedding.embed(sentence)
#         embed = sentence.embedding.detach().tolist()
#
#         tmp_data[idx] = embed
#         pd_tdata = pd.DataFrame(tmp_data)
#         # print(sentence.embedding)
#         print("E ",idx)
#
#
#     pd_tdata.to_pickle(text_emb_fname)
# else:
#     pd_tdata = load_pkl(text_emb_fname)
#     # X_train["text_embeding"] = tmptrain['tmptrain']



# In[ ]:



# In[ ]:


#X_train= []


from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
X_test_table = X_test['text_embeding']
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
f1_score(y_test, y_pred)

