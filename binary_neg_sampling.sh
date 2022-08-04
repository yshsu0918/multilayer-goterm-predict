for ARG in $@; do
echo ${ARG}
python3 binary_neg_sampling.py \
          --epoch 10 \
          --train 0 \
          --pickle_train ../Dataset/D_FSHH/D_FSHHneg10_${ARG}_train.pickle \
          --pickle_valid ../Dataset/D_FSHH/D_FSHHneg10_${ARG}_valid.pickle \
          --pickle_test ../Dataset/D_FSHH/D_FSHHneg10_${ARG}_test.pickle \
          --result_auc ./result/0511_binary_neg10_${ARG}_auc.png \
          --model_path ./net/0430_binary_${ARG}_neg10.pt \
          --csv_path ./log/0430_${ARG}.csv
    
done