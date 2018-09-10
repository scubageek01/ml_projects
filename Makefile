.PHONY: all clean test

include .env

all: data/raw/pima.csv data/raw/iris.csv data/raw/housing.csv data/processed/dataframe reports/figures/classification_model_comparison_results.txt reports/figures/regression_model_comparison_results.txt

clean:
	rm -rf data/raw/*.csv
	rm -rf data/processed/*.npy
	rm -rf models/*.model
	rm -rf reports/figures/*.png
	rm -rf reports/figures/*.txt
	find . -name '*.pyc' -exec rm -f {} +

data/raw/iris.csv:
	python src/data/download.py ${IRIS_URL} $@

data/raw/housing.csv:
	python src/data/download.py ${HOUSING_URL} $@

data/raw/pima.csv:
	python src/data/download.py ${PIMA_URL} $< $@ 

data/processed/dataframe: data/raw/iris.csv
	python src/data/preprocess.py $< $@

reports/figures/classification_model_comparison_results.txt:
	python src/evaluation/evaluate_classification.py $< $@

reports/figures/regression_model_comparison_results.txt:
	python src/evaluation/evaluate_regression.py $< $@

tune_knn:
	python src/evaluation/tune_knn.py

test: all
	pytest src


