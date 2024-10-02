import reflex as rx

import pandas as pd
from typing import Union, List, Tuple
import re
from nltk.tokenize import word_tokenize
from collections import defaultdict, Counter
import itertools

emails = pd.read_csv('emails.csv')
my_corpus = emails.iloc[:50,:]

import sys
sys.path.append('/Users/alina/NGramModel/')
from rxconfig import config

class PrefixTreeNode:

    def __init__(self):

        self.children = {}
        self.is_end_of_word = False 

class PrefixTree:
    def __init__(self, vocabulary):

        self.root = PrefixTreeNode()
        for word in vocabulary:
            self._insert(word)

    def _insert(self, word: str):

        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = PrefixTreeNode()
            node = node.children[char]
        node.is_end_of_word = True

    def _collect_words(self, node, prefix, results):

        if node.is_end_of_word:
            results.append(prefix)
        for char, child_node in node.children.items():
            self._collect_words(child_node, prefix + char, results)

    def search_prefix(self, prefix):

        node = self.root

        for char in prefix:
            if char not in node.children:
                return [] 
            node = node.children[char]
        
   
        results = []
        self._collect_words(node, prefix, results)
        return results

class WordCompletor:
    def __init__(self, corpus):

        self.word_frequency = Counter(list(itertools.chain.from_iterable(corpus))) 

        vocabulary = list(self.word_frequency.keys())
    
        self.prefix_tree = PrefixTree(vocabulary)

    def get_words_and_probs(self, prefix: str):


        words = self.prefix_tree.search_prefix(prefix)
        
        total = sum(self.word_frequency.values())
        
        probs = [self.word_frequency[word]/total if total > 0 else 0 for word in words]

        return words, probs
    

class NGramLanguageModel:
    def __init__(self, corpus, n):

        self.n = n
        self.ngram_next_counts = defaultdict(Counter)
        self.ngram_counts = Counter()
        
        for sent in corpus:
            ngrams, ngrams_w_next = self.make_ngrams(sent, n)
            for ngram, ngram_next in zip(ngrams, ngrams_w_next):
                self.ngram_next_counts[ngram][ngram_next[-1]] += 1
                self.ngram_counts[ngram] += 1


    def make_ngrams(self, sentence, n):

        sentence = (n-1) * ['<PAD>'] + sentence + ['.']
        ngrams = []
        ngrams_w_next = []

        for i in range(n - 1, len(sentence)-1):
            preceding = sentence[i - n + 1:i+1] 
            word = sentence[i+1]
            ngrams_w_next.append(tuple([*preceding, word]))
            ngrams.append(tuple(preceding))
        
        return ngrams, ngrams_w_next
    
    def get_next_words_and_probs(self, prefix):

        if len(prefix) <= self.n-1:
            prefix_tuple = tuple(prefix)
        else:
            prefix_tuple = tuple(prefix[-(self.n):])

        if prefix_tuple not in self.ngram_counts:
            return [], []
        
        next_word_counts = self.ngram_next_counts[prefix_tuple]
        total_count = self.ngram_counts[prefix_tuple]
        
        next_words = [word for word in next_word_counts.keys()]
        probs = [count/total_count for count in next_word_counts.values()]
        
        return next_words, probs

class TextSuggestion:
    def __init__(self, word_completor, n_gram_model):
        self.word_completor = word_completor
        self.n_gram_model = n_gram_model

    def suggest_text(self, text, n_words=3, n_texts=1):

        suggestions = []

    
        string_options, string_probs = self.word_completor.get_words_and_probs(text[-1])


        string_zip = sorted(zip(string_probs, string_options), reverse = True)
        
        if len(string_zip) > 0:

            best_string = string_zip[0][1]
        else:
            best_string = text[-1]

        text[-1] = best_string

        for i in range(n_words):
            
            sentence_options, sentence_probs = self.n_gram_model.get_next_words_and_probs(text)
            sentence_zip = sorted(zip(sentence_probs, sentence_options), reverse = True)

            if len(sentence_zip) > 0:
                best_sentence = ''
                best_sentence = sentence_zip[0][1]

            else:
                break
            text.append(best_sentence)

        suggestions.append(list(text[-(n_words+1):]))

        return suggestions


my_corpus['message_wo_metadata'] = my_corpus['message'].apply(lambda x: x.split('\n\n', 1)[1])



def split_forward(msg):
    
    if 'Subject:' in msg:
        return msg.rsplit('Subject:', 1)[-1].split('\n\n', 1)[-1]
        
    else:
        return msg
       
my_corpus['message_wo_forward'] = my_corpus['message_wo_metadata'].apply(lambda x: split_forward(x))
my_corpus['message_wo_meetings'] = my_corpus['message_wo_forward'].apply(lambda x: x.split('----------------------', 1)[0])
my_corpus['message_wo_subscription_metadata'] = my_corpus['message_wo_meetings'].apply(lambda x: split_forward(x))


my_corpus['message_wo_emails'] = my_corpus['message_wo_subscription_metadata'].apply(lambda x: re.sub(r'\S*@\S*\s?', '', x))
my_corpus['message_wo_urls'] = my_corpus['message_wo_emails'].apply(lambda x: re.sub(r'http\S+', '', x))
my_corpus['message_wo_files'] = my_corpus['message_wo_urls'].apply(lambda x: re.sub(r'-\s\S*.\S*', '', x))

def preprocess(x):

    x_lowercase = x.lower()
    x_no_digits = re.sub('\d+', '', x_lowercase)
    x_no_nextstring = x_no_digits.replace('\n', ' ')
    x_no_punctuation = re.sub(r'[^a-zA-Z\s]', ' ', x_no_nextstring)
    x_final = re.sub(' +', ' ', x_no_punctuation)

    return x_final

my_corpus['message_preprocessed'] = my_corpus['message_wo_files'].apply(lambda x: preprocess(x))
my_corpus['message_tokenized'] = my_corpus['message_preprocessed'].apply(lambda x: x.split())

word_completor = WordCompletor(my_corpus.iloc[:,10])
n_gram_model = NGramLanguageModel(corpus=my_corpus.iloc[:,10], n=2)
text_suggestion = TextSuggestion(word_completor, n_gram_model)


class State(rx.State):

    prompt = ""
    suggested_text = ""
    suggested_text_2 = ""
    processing = False
    complete = False

    def get_suggestion(self):
        if self.prompt == "":
            return rx.window_alert("Please, enter the prompt!")
        self.processing, self.complete = True, False
        yield
        response = text_suggestion.suggest_text(list(self.prompt.split()), n_words=3, n_texts=1)
        self.suggested_text = ' '.join(response[0])
        self.processing, self.complete = False, True


def index():
    return rx.center(
        rx.vstack(
            rx.heading("Text suggestion online!", size = "8",),
            rx.heading(
            "Made by Grushina Daria, HSE-NES Joint Programme",
            size="5",),
            rx.input(
                placeholder="Enter a prompt... ",
                on_blur=State.set_prompt,
                width="25em",
                border_color="#1c2024",
            ),
            rx.button(
                "suggest continuation", 
                on_click=State.get_suggestion,
                width="25em",
                loading=State.processing, 
                background_color="#1c2024"
            ),
            rx.cond(
                State.complete,
                rx.text(State.suggested_text, text_align="center", font_weight="bold", color="black")
            ),
            align="center",
        ),
        width="100%",
        height="100vh",
        background="linear-gradient(to right, #a8c0ff, #3f2b96)"
    )

app = rx.App()
app.add_page(index, title="Text suggestion online!")