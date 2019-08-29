'''Contains functions for data handling, cleaning and preprocessing'''

import pandas as pd
import numpy as np
import nltk
import sys

if not sys.platform.startswith('win'):
    import aspell

from imageio import imwrite as imsave
from wordcloud import WordCloud
from PyQt5.QtWidgets import QMessageBox

def clean(self, threshold=0.5):
    str = "Data cleaning results:\n\n"
    print('Cleaning data . . .')
    print('Original shape: ', self.df.shape)
    str = str + 'Original shape: {}\n'.format(self.df.shape)
    self.df.drop_duplicates(keep=False, inplace=True)  # remove duplicate entries
    print('Shape after dropping duplicates: ', self.df.shape)
    str = str + 'Shape after dropping duplicates: {}\n'.format(self.df.shape)

    # drop rows with too many missing values
    self.df.drop(self.df[(np.sum(self.df.isna(), axis=1) / self.df.shape[1] > threshold)].index, inplace=True)
    print('Shape after dropping rows with too many missing values: ', self.df.shape)
    str = str + 'Shape after dropping rows with too many missing values: {}\n'.format(self.df.shape)

    # drop columns with too many missing values
    self.df.drop(self.df.columns[np.sum(self.df.isna(), axis=0) / self.df.shape[0] > threshold], axis=1, inplace=True)
    print('Shape after dropping columns with too many missing values: ', self.df.shape)
    str = str + 'Shape after dropping columns with too many missing values: {}\n'.format(self.df.shape)

    for col in self.df.columns:
        if self.df[col].dtype == 'object':
            self.df.loc[self.df[col].isna(), col] = self.df[col].value_counts().index[0]
        else:
            self.df.loc[self.df[col].isna(), col] = np.mean(self.df[col])
    print('Number of NULL values remaining after imputation: ', np.sum(self.df.isna()))

    msgBox = QMessageBox()
    msgBox.setText(str)
    msgBox.setWindowTitle("Data cleaning results")
    msgBox.exec_()

class ReviewData:
    def __init__(self, filename):
        try: self.data = pd.read_csv(filename, delimiter=",", encoding='latin-1')
        except:
            try:
                self.data = pd.read_csv(filename, delimiter=";", encoding='latin-1')
            except:
                self.data = pd.read_csv(filename, encoding='latin-1')


    def store_text(self, textColumn):
        self.text = self.data[textColumn]

    def process_labels(self, labelsColumn):
        self.labelled_indices = self.data[labelsColumn].notna()
        self.labelled_data = self.data[self.labelled_indices]
        self.unlabelled_data = self.data[self.data[labelsColumn].isna()]

        self.labels = self.labelled_data[labelsColumn]

    def store_additional_fields(self, qual_profiles, quant_profiles, date = None, locations=None, loc=None):
        self.locations = locations
        self.loc = loc
        self.date = date

        self.quantitative_profiles = quant_profiles
        self.qualitative_profiles = qual_profiles
        self.profiles = []
        self.profiles.extend(quant_profiles)
        self.profiles.extend(qual_profiles)

        self.df = self.data

        selected_cols = []
        selected_cols.extend(self.profiles)

        if date != 'None' and date not in selected_cols:
            selected_cols.extend([date])

        if loc != 'None' and loc not in selected_cols:
            selected_cols.extend([loc])

        if locations[0] != 'None' and locations[1] != None:
            if self.df[locations[0]].dtype == 'object':
                # assume 0 is town and 1 is province
                location_data = pd.read_csv('./data/SouthAfricanCities.csv', encoding="ISO-8859-1")
                if locations[0] not in selected_cols:
                    selected_cols.extend([locations[0]])
                    self.qualitative_profiles.extend([locations[0]]) #TODO: why am I doing this again?
                if locations[1] not in selected_cols:
                    selected_cols.extend([locations[1]])
                    self.qualitative_profiles.extend([locations[1]])

                df = self.df[selected_cols]
                self.df = pd.merge(df, location_data[['AccentCity', 'ProvinceName', 'Latitude', 'Longitude']],
                                   how='inner', left_on=[locations[0], locations[1]],
                                   right_on=['AccentCity', 'ProvinceName']).drop(['AccentCity', 'ProvinceName'], axis=1)
                self.locations = ['Latitude', 'Longitude']
            else:
                if locations[0] not in selected_cols and locations[1] not in selected_cols:
                    selected_cols.extend(locations)
                    self.df = self.df[selected_cols]
        else:
            self.df = self.df[selected_cols]

    def create_wordcloud(self, text):
        wordcloud = WordCloud(mode = "RGBA",width = 1500, height=500, max_words=1000, stopwords=[]).generate(text)
        path = "graphics/wordcloud.png"
        imsave(path, wordcloud)
        return path

    def tokenise(self):
        self.unfiltered_tokens = []

        for i in range(len(self.text)):
            self.unfiltered_tokens.append(nltk.word_tokenize(self.text[i]))

        self.unfiltered_tokens = np.array(self.unfiltered_tokens)
        self.tokens = self.unfiltered_tokens #in case no filtering is done

    def get_types(self):
        all_tokens = [token for review in self.tokens for token in review]
        self.tokens_dist = nltk.FreqDist(all_tokens)
        self.types = set(token for review in self.tokens for token in review)

    def remove_stopwords(self, stop_words):
        self.tokens = []
        for review in self.unfiltered_tokens:
            i = 0
            placeholder = []
            for token in review:
                if token.lower() not in stop_words:
                    placeholder.append(token)
            self.tokens.append(placeholder)
            i = i + 1
        self.tokens = np.array(self.tokens)

    def stem(self, algorithm):
        if algorithm == 'Lancaster':
            stemmer = nltk.stem.LancasterStemmer()
        if algorithm == 'Porter':
            stemmer = nltk.stem.PorterStemmer()
        if algorithm == 'Snowball':
            stemmer = nltk.stem.snowball.EnglishStemmer()

        for review in self.tokens:
            for i in range(len(review)): #for each token
                if review[i][0] == "_": #special character
                    pass
                else:
                    if algorithm == 'Lemmatisation':
                        stemmer = nltk.stem.WordNetLemmatizer()
                        review[i] = stemmer.lemmatize(review[i])  # replace token with its lemma
                    else:
                        review[i] = stemmer.stem(review[i]) #replace token with stemmed version

    def spellcheck(self, spellchecker, num, exclude_list=None):
        if exclude_list != None:
            exclude_list = exclude_list.split(',')

        if spellchecker == 'aspell':
            s = aspell.Speller('lang', 'en')
            for review in self.tokens:
                for i in range(len(review)):
                    token = review[i].encode('utf8')
                    if sum(char.isdigit() for char in review[i])/len(review[i]) > 0.5:
                        # dont use encode/decode because then everything is an int
                        if num == True:
                            if exclude_list == None or review[i] not in exclude_list:
                                review[i] = '_num'  # encode to same token indicating a number is present
                    elif s.check(token) == False:
                        suggestions = s.suggest(token)
                        if any([word.lower() == token.decode('utf8').lower() for word in suggestions]):
                            review[i] = next((word for word in suggestions if word.lower() == token.decode('utf8').lower()), None)
                        else:
                            frequencies = np.array((self.tokens_dist.freq(token.decode(
                                'utf8')) * self.tokens_dist.N() - 1))  # exclude this instance from the frequency count
                            for j in range(len(suggestions)):
                                frequencies = np.append(frequencies, self.tokens_dist.freq(
                                    suggestions[j]) * self.tokens_dist.N())
                            most_frequent_index = np.argmax(frequencies)
                            if most_frequent_index != 0:
                                review[i] = suggestions[most_frequent_index - 1]
        elif num == True:
            for review in self.tokens:
                for i in range(len(review)):
                    if sum(char.isdigit() for char in review[i]) / len(review[i]) > 0.5:
                        if exclude_list == None or review[i] not in exclude_list:
                            review[i] = '_num'  # encode to same token indicating a number is present

    def make_lowercase(self):
        for review in self.tokens:
            for i in range(len(review)):
                review[i] = review[i].lower()

    def store_as_text(self):
        self.cleantext = []
        for review in self.tokens:
            self.cleantext.append(" ".join(token for token in review))
        self.cleantext = np.array(self.cleantext)


class ClientData:
    def __init__(self, filename):
        self.df = pd.read_csv(filename)

    def save_relevant_fields(self, links, qual_profiles=None, quant_profiles=None):
        self.links = links
        self.quantitative_profiles = quant_profiles
        self.qualitative_profiles = qual_profiles
        self.profiles = []
        self.profiles.extend(quant_profiles)
        self.profiles.extend(qual_profiles)

        selected_cols = []
        selected_cols.append(links['clients_side'])
        selected_cols.extend(self.profiles)

        self.df = self.df[selected_cols]
        self.df = self.df.rename(columns={links['clients_side']: 'Client_num'})

    def clean(self, threshold = 0.5):
        clean(self, threshold=threshold)
