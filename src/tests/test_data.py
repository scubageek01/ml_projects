from src.data import read_raw_data, preprocess_data, get_features, get_response


def test_raw_shape():
    dataframe = read_raw_data('data/raw/pima.csv')
    assert dataframe.shape == (768, 9)


def test_feature_and_response_shape():
    dataframe = read_raw_data('data/raw/pima.csv')
    processed = preprocess_data(dataframe)
    processed = processed.values
    features = get_features(processed)
    response = get_response(processed)

    assert features.shape == (768, 8)
    assert response.shape == (768,)
