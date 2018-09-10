from sklearn.externals import joblib


def main():
    model = joblib.load('models/knn.model')

    oos_data = [[3, 5, 4, 2], [6,3,4,1.3], [6.,3.,6.7,3], [4.9,3.1,1.5,0.1]]

    results = model.predict(oos_data)
    print(results)


if __name__ == '__main__':
    main()