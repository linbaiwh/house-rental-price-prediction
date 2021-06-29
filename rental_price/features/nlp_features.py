import logging
import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import Normalizer, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.decomposition import NMF
from sklearn.pipeline import Pipeline

from textblob import TextBlob

logger = logging.getLogger(__name__)
nlp = spacy.load('en_core_web_md')

# * Named Entity Analysis
def find_named_entity(text):
    """Generate named entities for text, using spaCy.

    Args:
        text: the text of a document

    Returns:
        Dict: dictionary of contained named entities
    """
    entities = {}

    doc = nlp(text)
    for ent in doc.ents:
        entities[ent.label_] = entities.get(ent.label_, []) + [ent.text]
    
    return entities

def named_entity_sum(X):
    """Generate table to summarize the named entities in the corpus

    Args:
        X (pandas.Series): corpus in the format of pandas.Series

    Returns:
        pandas.DataFrame: each column represents words under a type of named entities 
    """
    entities = X.map(find_named_entity)
    df = pd.DataFrame(entities.tolist())
    entity_sum = {}
    for column in df.columns:
        entity_sum[column] = df[column].explode().drop_duplicates().sort_values()

    return pd.DataFrame.from_dict(entity_sum, orient='index').transpose()

