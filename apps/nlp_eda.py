from apps_context import read_review, save_to_feature, save_to_processed, save_model, read_model

import rental_price.features.nlp_features as nlp_feat

vect_params = {
    'lowercase': False,
    'max_df': 0.8,
    'ngram_range': (1,2),
    'stop_words': ['and', 'as', 'the', 'to', 'of', 'in', 'on', 'it', 'that', 'this', 
    'an', 'a', 'for', 'at', 'with', 'from', 'we', 'my', 'they', 'have', 'you', 'our', 
    'so', 'but', 'there', 'do', 'make', 'get', 'be', 'or']
}

dc_params = {
    'n_components': 10
}

def preprocess(df, text_col, tokenizer=nlp_feat.tokenizer_lemma_ent, entity=False, save=False):
    docs = nlp_feat.gen_en_docs(df[text_col])
    if entity:
        entity_df = nlp_feat.named_entity_sum(docs)
        save_to_feature('named_entities', entity_df)
    
    tokens = nlp_feat.gen_tokens(docs, tokenizer)
    if save:
        save_to_processed('review_tokens', tokens)
    return tokens

def top_words(tokens, **kwargs):
    df = nlp_feat.get_top_n_words(tokens, **kwargs)
    return nlp_feat.plot_top_words(df, 'top_words', 'top_words_freq')


def topics_modeling(tokens, decompose=nlp_feat.TruncatedSVD, num_topics=10, **kwargs):
    _, model, feature_words = nlp_feat.topics_decompose(tokens, decompose, **kwargs)
    return nlp_feat.plot_topic_words(model, feature_words, 20, 'Topics Modeling', num_topics=num_topics)


def nmf_num_topics_compare(tokens, nums_topics):

    for num in nums_topics:
        dc_nmf_param = {
            'init': 'nndsvd',
            'n_components': num
        }

        nmf_params = {
            'scaler': nlp_feat.Normalizer,
            'vect_params': vect_params,
            'dc_params': dc_nmf_param
        }

        topic_fig, topic_df = topics_modeling(tokens, decompose=nlp_feat.NMF, num_topics=num, **nmf_params)
        save_to_feature(f'NMF_{num}_Topics', df=topic_df, pic=topic_fig)



if __name__ == '__main__':
    # df_reviews_raw = read_review(nrows=60000)
    # tokens = preprocess(df_reviews_raw, 'comments', entity=True, save=True)

    # top_words_fig = top_words(tokens, n=30, **vect_params)
    # save_to_feature('top_30_words', pic=top_words_fig)

    tokens = read_review(processed=True, nrows=40000)['comments']
    tokens = tokens.map(nlp_feat.normalize_text)

    dc_nmf_param = {
        'init': 'nndsvd',
        'n_components':8
    }

    nmf_params = {
        'scaler': nlp_feat.Normalizer,
        'vect_params': vect_params,
        'dc_params': dc_nmf_param
    }

    pipe, _, _ = nlp_feat.topics_decompose(tokens, nlp_feat.NMF, **nmf_params)
    save_model('NMF_8_Topics', pipe)

    # pipe = read_model('NMF_8_Topics')
    topics_list = ['Community', 'Policy', 'Location', 'House', 'Willing', 'Host_1', 'Host_2', 'Recommend']
    test_tokens = read_review(processed=True, skiprows=50000, nrows=20, names=['idx', 'comments'])['comments']
    test_tokens = test_tokens.map(nlp_feat.normalize_text)
    test_topics = nlp_feat.topics_transformer(test_tokens, pipe, topics_list)
    save_to_feature('test_8_topics', df=test_topics)


    # nums_topics = range(6, 11, 1)
    # nmf_num_topics_compare(tokens, nums_topics)

    # dc_nmf_param = {
    #     'init': 'nndsvd'
    # }

    # nmf_params = {
    #     'scaler': nlp_feat.Normalizer,
    #     'vect_params': vect_params,
    #     'dc_params': {**dc_params, **dc_nmf_param}
    # }

    # dc_lda_param = {
    #     'learning_method': 'online'
    # }

    # lda_params = {
    #     'tfidf': False,
    #     'vect_params': vect_params,
    #     'dc_params': {**dc_params, **dc_lda_param}
    # }

    # svd_params = {
    #     'vect_params': vect_params,
    #     'dc_params': dc_params
    # }

    # topic_fig, _ = topics_modeling(tokens, **svd_params)
    # save_to_feature('topics_LSA', pic=topic_fig)
    
    # topic_fig, _ = topics_modeling(tokens, decompose=nlp_feat.NMF, **nmf_params)
    # save_to_feature('topics_NMF', pic=topic_fig)
    
    # topic_fig, _ = topics_modeling(tokens, decompose=nlp_feat.LatentDirichletAllocation, **lda_params)
    # save_to_feature('topics_LDA', pic=topic_fig)
