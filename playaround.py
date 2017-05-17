import csv
from gensim.models import Word2Vec
from itertools import chain
from nltk import tokenize
from collections import Counter
import pickle

## helpers

# serilaze obj to file
def save_obj(obj, file):
    pickle.dump(obj, open(file, "wb"))

# load obj from file
def read_obj(file):
    return pickle.load(open(file,"rb"))

## preare data
def prepare_train_data():
    with open('train.csv') as file:
        reader = csv.reader(file)
        next(reader)
        data = [[int(row[0]), int(row[1]), int(row[2]), row[3], row[4], int(row[5])] for row in list(reader)]
        return data

def prepare_test_data():
    with open('test.csv') as file:
        reader = csv.reader(file)
        next(reader)
        data = [[int(row[0]), row[1], row[2]] for row in list(reader)]
        return data

def save_tokenized_train_data():
    save_obj(prepare_train_data(),"train_data_tokenized")

def load_train_data():
    return read_obj("train_data")

def load_test_data():
    return read_obj("test_data")


data = load_train_data()

totaol_num = len(data)
correct_data = [pair for pair in data if pair[5] ==1]
correct_num = len(correct_data)


test_data = prepare_test_data()

ALL = -1
NOT_CORRECT = 0
CORRECT = 1



def question_to_sents(question):
    return [[word for word in tokenize.word_tokenize(sent)] for sent in tokenize.sent_tokenize(question)]

def tokenize_train_data_remain_structure(train_data=data):
    q1_set = [question_to_sents(question.lower()) for question in [row[3] for row in train_data]]
    q2_set = [question_to_sents(question.lower()) for question in [row[4] for row in train_data]]
    for row in train_data:
        row.append(question_to_sents(row[3].lower()))
        row.append(question_to_sents(row[4].lower()))
    return train_data

def save_tokenize_train_data_remain_structure(train_data=data):
    tokenized_train_data = tokenize_train_data_remain_structure(train_data)
    save_obj(tokenized_train_data,"train_data")






def data_contains_subtext(subtext, category=None):
    if category:
        return [(pair[0],pair[3],pair[4],pair[5]) for pair in data if (subtext in pair[3] or subtext in pair[4]) and pair[5] == category]
    else:
        return [(pair[0],pair[3],pair[4],pair[5]) for pair in data if (subtext in pair[3] or subtext in pair[4])]

def test_data_contains_subtext(subtext):
    return [pair for pair in test_data if (subtext in pair[1] or subtext in pair[2])]




# output correct answer pairs into a sample file
def write_correct_to_csv(correct_data=correct_data, start=0, count= 100, filename= "sample_train_correct.csv"):
    if start + count < len(correct_data):
        end = start + count
    else:
        end = len(correct_data)
    with open(filename,'w') as file:
        writer = csv.writer(file)
        for row in correct_data[start:end]:
            writer.writerow(row)



## test text features



# find the diff length of correct answer in train set
def diff_length_of_correct_pair_in_train(train_data=data):
    return [(abs(len(row[6]) - len(row[7])),row[0]) for row in train_data if row[5] == 1]

[(row[0],row[3],row[4]) for row in data if row[5] == 1 and abs(len(row[6]) - len(row[7])) == 3]


[(row[0],row[3],row[4]) for row in data if row[5] == 1 and abs(len(row[6]) - len(row[7])) == 0]



## word2vec model
def train_word2vec_model(sents):
    return Word2Vec(sents,min_count=5,iter=5)

def save_model(model):
    model.save("word2vec_train_data")

def question_set_to_sents(q_set):
    return [[word for word in tokenize.word_tokenize(sent)] for question in q_set for sent in tokenize.sent_tokenize(question)]

def tokenize_train_data(train_data=data):
    q1_set = [question.lower() for question in [row[3] for row in train_data]]
    q2_set = [question.lower() for question in [row[4] for row in train_data]]
    return [*question_set_to_sents(q1_set), *question_set_to_sents(q2_set)]

def prepare_word2vec_data():
    sents = tokenize_train_data()
    save_obj(sents,"train_sents")




import spacy
from spacy.symbols import *
import itertools as it
nlp = spacy.load("en")

def first_verb(sent):
    for w in sent:
        if w.pos == VERB:
            return w
def first_to(sent):
    for w in sent:
        if w.tag_ == "TO":
            return w

def is_what_is_to(sent):
    if sent[0].tag_ == "WP":
        fv = first_verb(sent)
        if fv and fv.lemma_ == "be" and first_to(sent):
            return True
    else:
        return False

def show_meta(sent,type="tag"):
    sent = nlp(sent)
    if type == "tag":
        return [(w.orth_, w.tag_) for w in sent]
    elif type == "p": #phrase
        return list(sent.noun_chunks)
    elif type == "pos": #phrase
        return [(w.orth_, w.pos_) for w in sent]



sents = ((nlp(pair[3]),pair) for pair in data)
sample = [ (sent[0],sent[1][4]) for sent in it.islice((sent for sent in sents if is_what_is_to(sent[0]) and sent[1][5] == 1),0,100)]


sum(1 for sent in (sent for sent in sents if is_what_is_to(sent[0]) and sent[1][5] == 1))
