import pytest
import pandas as pd

from rental_price.features import nlp_features as nlp_feat

corpus = [
    """Welcome to "The Mission," the sunniest neighborhood in town. Enjoy world-class dining, shopping, and nightlife on Valencia, or a Tartine picnic in Dolores Park, all just steps from our 100 year old home.  We've been Hosts since Airbnb's 1st year (2009), and can suggest restaurants or insider sight-seeing ideas. We love our home, our hometown, and our neighborhood, and we look forward to sharing them with you. PLEASE click below to read more about this private room and bath in a shared home. Your sunny room looks out over a lush garden, and includes all the comforts of home... Queen bed, sofa, dresser, closet, and a rocking chair.  Just up the hall, your private bathroom has a "like rain" shower head over a clawfoot tub. (The bathroom is not "en-suite.") For a third guest (or for two guests who want separate beds) there is a full second bedroom available. ($60/night, and only available if you book Mission Sunshine.) You'll have your own bedroom and private bathroom, as well as shared ac"""
, """For your health & safety, we are implementing enhanced expert-led cleaning protocols. We are taking every precaution and making every effort to bring you peace of mind while you stay with us. Our cleaning staff will be trained to follow the recommended enhanced cleaning protocols and sanitation techniques. Airbnb has developed. We will be including, besides, basic staples, supplies to enhance your safety in and out of the house. (Hand sanitizer, disinfectant(s), alcohol, masks, etc.) We refurbished the entire flat last October! It can be done; Edwardian bones with mid-century furnishings! Furnished/ Modernized Victorian Flat in quiet one-block street moments away from transportation (MUNI Castro Sta.,  F Line-Vintage Cars), etc. Walking & Public Transit Scores.: 98-100 1. Welcome to our Edwardian Flat / Long Term Rental Property centrally located in quiet street moments away from the Castro District, Mission Dolores, Dolores Park, Church Street, the Valencia Corridor, Cafes, Restaurant"""
, """This private studio apartment has two beds, private bath in a superb location includes access to courtyard and private sauna.  Within a block of Castro's nightlife where 9 transit lines converge taking you anywhere in the city. If you crave privacy and independence, then our comfy studio will work out well for you.  You have your own entrance with complete privacy from the rest of the house. This studio apartment is in an unbeatable location only a block from Castro and 18th Street and still very quiet and secluded.  A comfortable studio room with two beds, private entrance, private bath, refrigerator, microwave, TV, DVD player steps from 18th and Castro, the center of the Castro Business District.   Included are laundry facilities, wifi and access to a courtyard with water features and in home sauna. Shared courtyard.  Private sauna.  Internet (wifi), washer and dryer and satellite TV. You will have complete privacy.  We are available to answer your questions.  This is the ideal place"""
, """Stay in an upper-floor suite with private bath in a bright, charming Marina-style historic home. West Portal is a safe, peaceful "transit village" with cafes & restaurants. Walk 3 mins to the metro (subway) station. Wait 2-4 mins for train. Ride downtown in 15 mins! Suite directly accessible from front door, and feels private from rest of house.  But you also have easy access to living room & kitchen (~1000 sq ft / 100mÂ²). AMENITIES - Private renovated bathroom - Coffee, tea, creamer, soymilk, cereal - Hairdryer, body wash, shampoo, conditioner, towels - Iron & board upon request There's a queen bed for two.  Or, ask for an air mattress. SHARED AREAS - Views of Mt Davidson, Forest Hill, & Mt Sutro - Open kitchen with new granite top and cherry cabinets - New stainless dishwasher, refrigerator, and gas stove EASY CHECK-IN with keypad entry system.  No physical keys needed. Self check-in is possible, giving you privacy & flexibility in arrival time. JUST STEPS from award-winning restaura"""
, """Excellent location!!  19th St between Valencia and Guerrero. Close to amazing restaurants, shops, public transport, playground, pool, grocer...you name it, it's right around the corner. Sunny two bedroom apartment with long sectional sofa in the heart of the Mission. Hardwood floors, laundry, large kitchen, separate dining room, modern decor. WiFi, TV (no cable), turntable, gas stove, quiet. Entire apartment is accessible to guests except for one bedroom closet and some drawers in the dresser. Guests have full use of hallway closet. I host only when I travel for a few weeks in the summer. If I am not available a local contact will be provided. The neighborhood is teeming with activity. Plenty of things to do and people to watch. Safe and easy. Great food, great drinks, suitable for children, and sunny!! One of the few neighborhoods where you can count on the sun. Very close to public transportation and easy to catch a cab or walk. Parking is difficult. This is my home where I live with"""
, """Die Unterkunft am Dolores Park ist so charmant, wie der Name es sagt. Es ist ein grosszÃ¼giges, liebevoll eingerichtetes Zuhause mit toller KÃ¼chenausstattung. Shelagh ist eine umsichtige, freundliche und grosszÃ¼gige Gastgeberin. Wir haben uns sehr wohl und willkommen gefÃ¼hlt. Es gibt auch ein wunderbares Deck, wo man grillieren und essen kann. Die Lage ist toll, das Zentrum ist in ca einer halben Stunde mit dem Muni (Linie J) zu erreichen. Gleichzeitig ist man in einem tollen Quartier untergebracht. Valencia Street, die gleich um die Ecke ist, ist toll, um sich wie ein Local zu fÃ¼hlen.  Es gibt viele kleine aber feine Lokale zum Essen und Shops. Ausserdem ist die Bi-Rite Creamery gleich um die Ecke. Diese hat zu gewissen Zeiten enorm lange Wartezeiten, weil das Eis da so toll ist."""
]


@pytest.fixture
def text_series():
    return pd.Series(corpus)

@pytest.fixture
def all_doc_series(text_series):
    return text_series.map(nlp_feat.nlp)

def test_detect_en(all_doc_series):
    assert nlp_feat.detect_en(all_doc_series[0]) == True
    assert nlp_feat.detect_en(all_doc_series[5]) == False
    print(all_doc_series[5]._.language['language'])

def test_gen_en_docs(text_series):
    docs = nlp_feat.gen_en_docs(text_series)
    assert docs.shape[0] == 5

@pytest.fixture
def doc_series(text_series):
    return nlp_feat.gen_en_docs(text_series)


def test_find_named_entity(doc_series):
    text = doc_series[0]
    entities = nlp_feat.find_named_entity(text)
    print(entities.keys())
    print(entities.values())
    assert len(entities.keys()) > 0

def test_named_entity_sum(doc_series):
    entity_sum = nlp_feat.named_entity_sum(doc_series)
    print(entity_sum.columns)
    print(entity_sum.head())
    assert entity_sum.shape[0] > 0
    assert entity_sum.shape[1] > 0

def test_tokenizer_lemma_ent(doc_series):
    doc = doc_series.iloc[0]
    tokens = nlp_feat.tokenizer_lemma_ent(doc)
    print(' '.join(tokens))
    assert len(tokens) > 0

def test_gen_tokens(doc_series):
    tokens = nlp_feat.gen_tokens(doc_series, nlp_feat.tokenizer_lemma_ent)
    assert isinstance(tokens.iloc[0], str) == True

@pytest.fixture
def tokens_series(doc_series):
    return nlp_feat.gen_tokens(doc_series, nlp_feat.tokenizer_lemma_ent)


def test_get_top_words(tokens_series):
    df = nlp_feat.get_top_n_words(tokens_series, n=30)
    assert df.shape[0] == 30
    assert df.columns.tolist() == ['top_words', 'top_words_freq']
    nlp_feat.plot_top_words(df, 'top_words', 'top_words_freq')


def test_sentiment_feature(tokens_series):
    df = nlp_feat.sentiment_feature(tokens_series)
    assert df.shape[0] == tokens_series.shape[0]
    assert df.columns.tolist() == ['text', 'polarity', 'subjectivity']

def test_topics_decompose(tokens_series):
    model, feature_names = nlp_feat.topics_decompose(tokens_series, nlp_feat.TruncatedSVD)
    assert model.components_.shape[0] == min(10, tokens_series.shape[0])
    assert model.components_.shape[1] == len(feature_names)

def test_plot_topic_words(tokens_series):
    model, feature_names = nlp_feat.topics_decompose(
        tokens_series, nlp_feat.TruncatedSVD, 
        dc_params={'n_components': 4}
        )
    _, topic_df = nlp_feat.plot_topic_words(model, feature_names, 10, 'Topics in LSA', num_topics=4)
    assert topic_df.shape[1] == 8
    assert topic_df.shape[0] == 10
    print(topic_df.head())


