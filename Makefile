.PHONY: all clean test

include .env

all: data/raw/pima.csv data/raw/iris.csv data/processed/dataframe reports/figures/model_comparison_results.txt tune_knn

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

data/processed/dataframe: data/raw/iris.csv
	python src/data/preprocess.py $< $@

reports/figures/model_comparison_results.txt:
	python src/evaluation/evaluate.py $< $@

tune_knn:
	python src/evaluation/tune_knn.py

test: all
	pytest src


