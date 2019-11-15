#!/bin/bash
DATA_DIR="/data/tungtv/Code/dataset/dataset_cafebiz_45" && \
python3 -m acr.preprocessing.acr_preprocess_cafebiz \
	--input_articles_csv_path ${DATA_DIR}/data-thang45-final.parquet \
 	--input_word_embeddings_path ${DATA_DIR}/word2vec/model.txt \
 	--vocab_most_freq_words 100000 \
	--max_words_length 1000 \
 	--output_word_vocab_embeddings_path ${DATA_DIR}/pickles/acr_word_vocab_embeddings.pickle \
 	--output_label_encoders ${DATA_DIR}/pickles/acr_label_encoders.pickle \
 	--output_tf_records_path "${DATA_DIR}/articles_tfrecords/cafebiz_articles_*.tfrecord.gz" \
	--output_articles_csv_path "${DATA_DIR}/cafebiz_articles.csv" \
 	--articles_by_tfrecord 1000
