import sys

sys.path.append('src')

from data.preprocess import read_raw_data, read_processed_data


def test_raw_shape():
    dataframe = read_raw_data('data/raw/pima.csv')
    assert dataframe.shape == (768, 9)


def test_feature_and_response_shape():
    features = read_processed_data('data/processed/features.npy')
    response = read_processed_data('data/processed/response.npy')

    assert features.shape == (768, 8)
    assert response.shape == (768,)
