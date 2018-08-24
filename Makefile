.PHONY: all clean test

include .env

all: data/raw/pima.csv data/processed/data.pickle models/lda.model models/logreg.model models/random_forest.model

clean:
	rm -rf data/raw/*.csv
	rm -rf data/processed/*.pickle
	rm -rf models/*.model
	rm -rf reports/figures/*.png
	find . -name '*.pyc' -exec rm -f {} +

data/raw/iris.csv:
	python src/data/download.py ${IRIS_URL} $@

data/raw/pima.csv:
	python src/data/download.py ${PIMA_URL} $< $@ 

data/processed/data.pickle: data/raw/pima.csv
	python src/data/preprocess.py $< $@ --features data/processed/features.pickle --response data/processed/response.pickle

models/logreg.model:
	python src/models/logistic_regression.py $< $@

models/lda.model:
	python src/models/lda.py $< $@

models/random_forest.model:
	python src/models/random_forest.py $< $@

test: all
	pytest src


