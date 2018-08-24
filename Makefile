.PHONY: all clean test

include .env

all: data/raw/pima.csv data/processed/data.pickle models/linear_discriminant_analysis.model models/logistic_regression.model models/random_forest.model

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

models/logistic_regression.model:
	python src/models/logistic_regression.py $< $@

models/linear_discriminant_analysis.model:
	python src/models/lda.py $< $@

models/random_forest.model:
	python src/models/random_forest.py $< $@

test: all
	pytest src


