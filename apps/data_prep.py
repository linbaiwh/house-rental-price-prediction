from apps_context import read_listing, save_to_feature, save_to_processed
import rental_price.utils.preprocess as preprocess
from rental_price.models.model_build import features_to_keep

def raw_prep():
    df = read_listing(processed=False, low_memory=False)
    df_norm = preprocess.drop_irr(df)
    df_norm['neighbourhood_group_cleansed'].value_counts()
    df_manhattan = preprocess.neighborhood_selection(df_norm)

    df_select = preprocess.col_selection(df_manhattan)
    df_select = preprocess.price_trf(df_select)
    df_select = preprocess.numeric_trf(df_select)
    df_select = df_select.loc[df_select['price'] <= 357.5]
    df_select = df_select[features_to_keep]
    df_select.info()

    save_to_processed('listing_detail', df_select, index=False)

if __name__ == '__main__':
    raw_prep()