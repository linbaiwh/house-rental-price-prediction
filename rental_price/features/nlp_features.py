import logging
import pandas as pd
import re
import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import Normalizer, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.decomposition import TruncatedSVD, NMF, LatentDirichletAllocation
from sklearn.pipeline import Pipeline
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from gensim.models.nmf import Nmf

import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)
nlp = spacy.load('en_core_web_md')
nlp.add_pipe("merge_entities")

@Language.factory("language_detector")
def create_language_detector(nlp, name):
    return LanguageDetector(language_detection_function=None)

nlp.add_pipe('language_detector')

def detect_en(doc):
    try:
        if doc._.language['language'] == 'en':
            return True
    except:
        return False
    return False

def gen_en_docs(text_series):
    docs = text_series.map(nlp, na_action='ignore')
    is_en = docs.map(detect_en)
    return docs.loc[is_en]

# * Named Entity Analysis
def find_named_entity(doc):
    """Generate named entities for text, using spaCy.

    Args:
        doc (str or spacy.Doc): the text of a document

    Returns:
        Dict: dictionary of contained named entities
    """
    entities = {}
    if isinstance(doc, str):
        doc = nlp(doc)
    for ent in doc.ents:
        entities[ent.label_] = entities.get(ent.label_, []) + [ent.text]
    
    return entities

def named_entity_sum(doc_series):
    """Generate table to summarize the named entities in the corpus

    Args:
        doc_series (pandas.Series of str or spacy.Doc): the series of text

    Returns:
        pandas.DataFrame: each column represents words under a type of named entities 
    """
    entities = doc_series.map(find_named_entity)
    df = pd.DataFrame(entities.tolist())
    entity_sum = {}
    for column in df.columns:
        entity_sum[column] = df[column].explode().drop_duplicates().dropna().sort_values().reset_index(drop=True)

    return pd.DataFrame.from_dict(entity_sum, orient='index').transpose()

def tokenizer_lemma_ent(doc, ents=True):
    if isinstance(doc, str):
        doc = nlp(doc)
    tokens = []
    for token in doc:
        if ents and token.ent_iob_ == 'B':
            if ents == True:
                tokens.append(token.ent_type_)
                continue
            elif token.ent_type_ in ents:
                tokens.append(token.ent_type_)
                continue
        if token.pos_ != 'PUNCT':
            if token.like_url:
                tokens.append('URL')
            elif token.like_email:
                tokens.append('EMAIL')
            elif token.like_num:
                tokens.append('NUM')
            elif not token.is_oov:
                tokens.append(token.lemma_)
    
    return tokens


def gen_tokens(doc_series, tokenizer):
    tokens = doc_series.map(tokenizer, na_action='ignore').dropna()
    tokens = tokens.map(' '.join)
    return tokens.map(normalize_text)


def normalize_text(text):
    text = re.sub(r'\b(he|she|PERSON)\b', 'host', text)
    return text

# * Frequency Analysis
def get_top_n_words(text_series, n=30, **kwargs):
    """Generate list of the n most frequent words in the corpus. Any transformation
    of the words are applied through tokenizer or other parameters of CounterVectorizer.

    Args:
        text_series (pandas.Series of str): the series of text
        n (int, optional): The number of most frequent words to present. Defaults to 100.

    Returns:
        List of string, List of float: 
            List of the n most frequent words, List of corresponding frequency
    """
    vectorizer = CountVectorizer(**kwargs)
    bag_of_words = vectorizer.fit_transform(text_series)
    mean_words = bag_of_words.mean(axis=0)
    word_freq = [(word, mean_words[0, idx] * 1000) for word, idx in vectorizer.vocabulary_.items()]
    word_freq.sort(key=lambda x: x[1], reverse=True)
    top_words, top_words_freq = list(zip(*word_freq[:n]))
    return pd.DataFrame(zip(top_words, top_words_freq), columns=['top_words', 'top_words_freq'])


def plot_top_words(df, words_col, freq_col):
    fig = plt.figure(figsize=(20,7))

    ax = sns.barplot(
        x=words_col,
        y=freq_col,
        data=df,
        palette='GnBu_d'
    )

    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        fontsize=14
    )

    plt.yticks(fontsize=14)
    plt.title('Top Words', fontsize=17)
    plt.show()
    return fig

# * Sentiment Analysis
def sentiment_feature(text_series):
    """Generate sentiment features for each sentence, using TextBlob package.

    Args:
        text_series (pandas.Series of str): the series of text

    Returns:
        DataFrame: DataFrame containing original text and sentiment features
    """
    blobs = text_series.map(TextBlob)
    df = pd.DataFrame()
    df['text'] = text_series
    df['polarity'] = blobs.map(lambda t: t.sentiment.polarity)
    df['subjectivity'] = blobs.map(lambda t: t.sentiment.subjectivity)
    return df

#   * Latent Semantic Analysis (LSA, LSI) & Latent Dirichlet Allocation (LDA)
def topics_decompose(text_series, decompose, scaler=Normalizer, tfidf=True, 
                        vect_params={}, dc_params={'n_components': 10}):
    """Topic modeling analysis

    Args:
        text_series (iterables): iterables of text
        decompose (sklearn.decomposition): the method of topic modeling.
        scaler (sklearn.preprocessing, optional): the scaler to be used before decomposition. Defaults to Normalizer.
        tfidf (bool, optional): Whether to use tfidf. Defaults to True.
        vect_params (dict, optional): parameters of CountVectorizer. Defaults to {}.
        dc_params (dict, optional): parameters of decompose. Defaults to {}.

    Returns:
        sklearn.decompose: fitted topic decomposition model
        list of str: list of words in the text_series
    """
    steps = [
        ('vect', CountVectorizer(**vect_params))
    ]
    if tfidf:
        steps.append(('tfidf', TfidfTransformer(use_idf=True, sublinear_tf=True)))

    if scaler == StandardScaler:
        steps.append(('scaler', StandardScaler(with_mean=False)))
    elif scaler == RobustScaler:
        steps.append(('scaler', RobustScaler(with_centering=False)))
    elif scaler is not None:
        steps.append(('scaler', scaler()))
    
    steps.append(('decompose', decompose(**dc_params)))
    pipe = Pipeline(steps).fit(text_series)
    feature_names = pipe.named_steps.vect.get_feature_names()

    return pipe, pipe.named_steps.decompose, feature_names 

def plot_topic_words(model, feature_names, n_top_words, title, num_topics=10):
    fig_row_num = num_topics // 5 + (num_topics % 5 > 0)
    fig_height = n_top_words // 4 * 2 * fig_row_num

    fig, axes = plt.subplots(fig_row_num, 5, figsize=(30, fig_height), sharex=True)
    axes = axes.flatten()
    topic_words = pd.DataFrame()

    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        topic_words[f'Topic_{topic_idx + 1}_words'] = top_features
        topic_words[f'Topic_{topic_idx + 1}_weights'] = weights

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f'Topic {topic_idx +1}', fontdict={'fontsize': 13})
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=9)
        for i in 'top right left'.split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=15)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.15)
    plt.show()
    return fig, topic_words

class coherence_score_calculation():
    def __init__(self, tokens_series):
        self.texts = tokens_series
        self.dictionary = None
        self.corpus = None

    def create_dictionary(self, **kwargs):
        self.dictionary = Dictionary(self.texts)
        self.dictionary.filter_extremes(**kwargs)
        return self

    def create_corpus(self):
        self.corpus = self.texts.map(self.dictionary.doc2bow).tolist()
    
    def coherence_nmf(self, num_topics=10, coherence='c_v', **kwargs):
        nmf = Nmf(
            corpus=self.corpus, 
            num_topics=num_topics, 
            id2word=self.dictionary, 
            **kwargs)
        cm = CoherenceModel(
            model=nmf, 
            texts=self.texts, 
            dictionary=self.dictionary, 
            coherence=coherence)
        return cm.get_coherence()


def topics_transformer(tokens_series, model, topics_list):
    transformed = model.transform(tokens_series)
    df = pd.DataFrame(transformed, columns=topics_list)
    df['texts'] = tokens_series
    return df


# amenities

def amenities_sum(df):
    amenities_vocab = [
        'kitchen', 'parking', 'alarm', 'hair dryer', 'iron',
        'coffee maker', 'fire extinguisher', 'crib', 'first aid', 'washer',
        'fireplace'
    ]
    amenity_vect = CountVectorizer(ngram_range=(1, 2), vocabulary=amenities_vocab, binary=True)
    amenities = amenity_vect.fit_transform(df['amenities'])
    return pd.DataFrame(amenities.todense(), columns=amenities_vocab)

