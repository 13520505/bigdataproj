#!/bin/bash
DATA_DIR="/home/tungtv/Documents/Code/News/dataset/dataset_cafebiz_mini_45" && \
python3 -m acr.acr_module \
    --path_pickle ${DATA_DIR}/pickles/acr_lable_encoders/ \
    --path_tf_record ${DATA_DIR}/articles_tfrecords/ \
 	--input_word_embeddings_path ${DATA_DIR}/word2vec/model.txt \
 	--vocab_most_freq_words 100000 \
	--max_words_length 1000 \
 	--output_word_vocab_embeddings_path ${DATA_DIR}/pickles/acr_word_vocab_embeddings/ \
 	--output_label_encoders ${DATA_DIR}/pickles/acr_lable_encoders/ \
 	--output_tf_records_path "${DATA_DIR}/articles_tfrecords/cafebiz_articles_*_@.tfrecord.gz" \
	--output_articles_csv_path ${DATA_DIR}/cafebiz_articles_csv/ \
 	--articles_by_tfrecord 1000 \
 	--mysql_host "localhost" \
 	--mysql_user "root" \
	--mysql_passwd "root" \
 	--mysql_database "database"
SCRIPT_PATH="/home/tungtv/Documents/Code/News/newsrecomdeepneural/acr_module/scripts/run_acr_training_cafebiz_local.sh"
. "$SCRIPT_PATH"

#