model_version='0011'
echo "train model: "${model_version}
python ../rnn.py --pred_domain age \
                 --batch_size 2048 \
                 --learning_rate 3e-4 \
                 --l2_reg 0 \
                 --num_lstm_units 128 \
                 --num_deep_rnn 1 \
                 --num_epoch 60 \
                 --skip_epoch 2 \
                 --skip_step 100 \
                 --hidden_layers 512,512,64 \
                 --model_version ${model_version}

model_version='0012'
echo "train model: "${model_version}
python ../rnn.py --pred_domain age \
                 --batch_size 2048 \
                 --learning_rate 3e-4 \
                 --l2_reg 0 \
                 --num_lstm_units 256 \
                 --num_deep_rnn 1 \
                 --num_epoch 60 \
                 --skip_epoch 2 \
                 --skip_step 100 \
                 --hidden_layers 512,512,64 \
                 --model_version ${model_version}

model_version='0013'
echo "train model: "${model_version}
python ../rnn.py --pred_domain age \
                 --batch_size 2048 \
                 --learning_rate 3e-4 \
                 --l2_reg 0 \
                 --num_lstm_units 256 \
                 --num_deep_rnn 1 \
                 --num_epoch 60 \
                 --skip_epoch 2 \
                 --skip_step 100 \
                 --hidden_layers 512,512,512,64 \
                 --model_version ${model_version}

model_version='0011'
echo "train model: "${model_version}
python ../rnn.py --pred_domain gender \
                 --batch_size 2048 \
                 --learning_rate 3e-4 \
                 --l2_reg 0 \
                 --num_lstm_units 128 \
                 --num_deep_rnn 1 \
                 --num_epoch 50 \
                 --skip_epoch 2 \
                 --skip_step 100 \
                 --hidden_layers 512,512,64 \
                 --model_version ${model_version}


