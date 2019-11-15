#!/bin/bash
#DATA_DIR="/home/minh/VCC/newsrecomdeepneural/chameleon/model" && \
DATA_DIR="/home/ngoc/Downloads/WORK_ADTECH/chameleon/nar_data" && \
JOB_PREFIX=cafebiz && \
JOB_ID=`whoami`_${JOB_PREFIX}_`date '+%Y_%m_%d_%H%M%S'` && \
MODEL_DIR='/home/ngoc/Downloads/WORK_ADTECH/chameleon/nar_data/output'${JOB_ID} && \
#MODEL_DIR='/home/minh/VCC/newsrecomdeepneural/chameleon/model/tmp/chameleon/jobs/'${JOB_ID} && \
echo 'Running training job and outputing to '${MODEL_DIR} && \
python3 -m nar.nar_trainer_cafebiz_full \
	--model_dir ${MODEL_DIR} \
	--train_set_path_regex "${DATA_DIR}/sessions_tfrecords_by_hour/cafebiz_sessions_hour_*.tfrecord.gz" \
	--train_files_from 0 \
	--train_files_up_to 3 \
	--training_hours_for_each_eval 1 \
	--save_results_each_n_evals 1 \
	--acr_module_resources_path "${DATA_DIR}/pickles/acr_articles_metadata_embeddings_45.pickle" \
	--nar_module_preprocessing_resources_path "${DATA_DIR}/pickles/nar_preprocessing_resources_45.pickle" \
	#--acr_module_resources_path ${DATA_DIR}/pickles/acr_articles_metadata_embeddings.pickle \
	#--nar_module_preprocessing_resources_path ${DATA_DIR}/pickles/nar_preprocessing_resources.pickle \
	--batch_size 64 \
	--truncate_session_length 20 \
	--learning_rate 0.0003 \
	--dropout_keep_prob 1.0 \
	--reg_l2 0.0001 \
	--softmax_temperature 0.2 \
	--recent_clicks_buffer_hours 1.0 \
	--recent_clicks_buffer_max_size 20000 \
	--recent_clicks_for_normalization 5000 \
	--eval_metrics_top_n 5 \
	--CAR_embedding_size 1024 \
	--rnn_units 10 \
	--rnn_num_layers 1 \
	--train_total_negative_samples 7 \
	--train_negative_samples_from_buffer 10 \
	--eval_total_negative_samples 7 \
	--eval_negative_samples_from_buffer 10 \
	--eval_negative_sample_relevance 0.1 \
	--enabled_articles_input_features_groups "category" \
	--enabled_clicks_input_features_groups "time,location" \
	--enabled_internal_features "recency,novelty,article_content_embeddings,item_clicked_embeddings" \
	--novelty_reg_factor 0.0

#--rnn_units 255 \
#--save_histograms
#--save_eval_sessions_negative_samples \
#--save_eval_sessions_recommendations \
#--disable_eval_benchmarks

#origin
#--enabled_articles_input_features_groups "category,author" \
#--enabled_clicks_input_features_groups "time,device,location,referrer" \


