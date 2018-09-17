python trainer.py --model esim \
 --model_config /notebooks/source/classifynet/model_config.json \
 --model_dir /data/xuht/test/classify_tianfeng_speech_command_word_dropout_focal_loss \
 --config_prefix /notebooks/source/classifynet/configs \
 --gpu_id 2 \
 --train_path "/data/xuht/tianfeng/speech_command_big.txt" \
 --dev_path "/data/xuht/duplicate_sentence/LCQMC/dev.txt" \
 --w2v_path "/data/xuht/Chinese_w2v/sgns.merge.char/sgns.merge.char.pkl" \
 --vocab_path "/data/xuht/tianfeng/emb_mat_word_dropout.pkl"

