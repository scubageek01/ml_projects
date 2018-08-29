.PHONY: all clean test

include .env

all: data/raw/pima.csv data/raw/iris.csv data/processed/data models/linear_discriminant_analysis.model models/logistic_regression.model models/random_forest.model reports/figures/model_comparison_results.txt

clean:
	rm -rf data/raw/*.csv
	rm -rf data/processed/*.npy
	rm -rf models/*.model
	rm -rf reports/figures/*.png
	rm -rf reports/figures/*.txt
	find . -name '*.pyc' -exec rm -f {} +

data/raw/iris.csv:
	python src/data/download.py ${IRIS_URL} $@

data/raw/pima.csv:
	python src/data/download.py ${PIMA_URL} $< $@ 

data/processed/data: data/raw/iris.csv
	python src/data/preprocess.py $< $@ --output_features data/processed/features --output_response data/processed/response

models/logistic_regression.model:
	python src/models/logistic_regression.py $< $@ --scaler minmax

models/linear_discriminant_analysis.model:
	python src/models/lda.py $< $@

models/random_forest.model:
	python src/models/random_forest.py $< $@

reports/figures/model_comparison_results.txt:
	python src/evaluation/evaluate_models.py $< $@

test: all
	pytest src


