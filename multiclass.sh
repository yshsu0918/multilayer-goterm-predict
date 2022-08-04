
echo '-------train---------'
python3 multiclass.py \
          --epoch 10 \
          --train 1 \
          --dim_in 8 \
          --label_size 5 \
          --pickle_train ../Dataset_3/bc/D_FSHH_ysctrnivlk_train.pickle \
          --pickle_valid ../Dataset_3/bc/D_FSHH_ysctrnivlk_valid.pickle \
          --pickle_test ../Dataset_3/bc/D_FSHH_ysctrnivlk_test.pickle \
          --model_path ./net/0622.pt
          #--csv_path ./log/0430_${ARG}.csv
echo '-------test---------'
python3 multiclass.py \
          --epoch 10 \
          --train 0 \
          --dim_in 8 \
          --label_size 5 \
          --pickle_train ../Dataset_3/bc/D_FSHH_ysctrnivlk_train.pickle \
          --pickle_valid ../Dataset_3/bc/D_FSHH_ysctrnivlk_valid.pickle \
          --pickle_test ../Dataset_3/bc/D_FSHH_ysctrnivlk_test.pickle \
          --model_path ./net/0622.pt
echo '-------test end---------'


echo '-------train---------'
python3 multiclass.py \
          --epoch 10 \
          --train 1 \
          --dim_in 8 \
          --label_size 6 \
          --pickle_train ../Dataset_3/bc/D_FSHH_ysctrnivlkot_train.pickle \
          --pickle_valid ../Dataset_3/bc/D_FSHH_ysctrnivlkot_valid.pickle \
          --pickle_test ../Dataset_3/bc/D_FSHH_ysctrnivlkot_test.pickle \
          --model_path ./net/0622.pt
          #--csv_path ./log/0430_${ARG}.csv
echo '-------test ot---------'
python3 multiclass.py \
          --epoch 10 \
          --train 0 \
          --dim_in 8 \
          --label_size 6 \
          --pickle_train ../Dataset_3/bc/D_FSHH_ysctrnivlkot_train.pickle \
          --pickle_valid ../Dataset_3/bc/D_FSHH_ysctrnivlkot_valid.pickle \
          --pickle_test ../Dataset_3/bc/D_FSHH_ysctrnivlkot_test.pickle \
          --model_path ./net/0622.pt
echo '-------test end---------'


echo '-------train---------'
python3 multiclass.py \
          --epoch 10 \
          --train 1 \
          --dim_in 10 \
          --label_size 5 \
          --pickle_train ../Dataset_3/bce/D_FSHH_ysctrnivlk_train.pickle \
          --pickle_valid ../Dataset_3/bce/D_FSHH_ysctrnivlk_valid.pickle \
          --pickle_test ../Dataset_3/bce/D_FSHH_ysctrnivlk_test.pickle \
          --model_path ./net/0622.pt
          #--csv_path ./log/0430_${ARG}.csv
echo '-------test---------'
python3 multiclass.py \
          --epoch 10 \
          --train 0 \
          --dim_in 10 \
          --label_size 5 \
          --pickle_train ../Dataset_3/bce/D_FSHH_ysctrnivlk_train.pickle \
          --pickle_valid ../Dataset_3/bce/D_FSHH_ysctrnivlk_valid.pickle \
          --pickle_test ../Dataset_3/bce/D_FSHH_ysctrnivlk_test.pickle \
          --model_path ./net/0622.pt
echo '-------test end---------'


echo '-------train---------'
python3 multiclass.py \
          --epoch 10 \
          --train 1 \
          --dim_in 10 \
          --label_size 6 \
          --pickle_train ../Dataset_3/bce/D_FSHH_ysctrnivlkot_train.pickle \
          --pickle_valid ../Dataset_3/bce/D_FSHH_ysctrnivlkot_valid.pickle \
          --pickle_test ../Dataset_3/bce/D_FSHH_ysctrnivlkot_test.pickle \
          --model_path ./net/0622.pt
          #--csv_path ./log/0430_${ARG}.csv
echo '-------test ot---------'
python3 multiclass.py \
          --epoch 10 \
          --train 0 \
          --dim_in 10 \
          --label_size 6 \
          --pickle_train ../Dataset_3/bce/D_FSHH_ysctrnivlkot_train.pickle \
          --pickle_valid ../Dataset_3/bce/D_FSHH_ysctrnivlkot_valid.pickle \
          --pickle_test ../Dataset_3/bce/D_FSHH_ysctrnivlkot_test.pickle \
          --model_path ./net/0622.pt
echo '-------test end---------'


echo '-------train---------'
python3 multiclass.py \
          --epoch 10 \
          --train 1 \
          --dim_in 12 \
          --label_size 5 \
          --pickle_train ../Dataset_3/bcep/D_FSHH_ysctrnivlk_train.pickle \
          --pickle_valid ../Dataset_3/bcep/D_FSHH_ysctrnivlk_valid.pickle \
          --pickle_test ../Dataset_3/bcep/D_FSHH_ysctrnivlk_test.pickle \
          --model_path ./net/0622.pt
          #--csv_path ./log/0430_${ARG}.csv
echo '-------test---------'
python3 multiclass.py \
          --epoch 10 \
          --train 0 \
          --dim_in 12 \
          --label_size 5 \
          --pickle_train ../Dataset_3/bcep/D_FSHH_ysctrnivlk_train.pickle \
          --pickle_valid ../Dataset_3/bcep/D_FSHH_ysctrnivlk_valid.pickle \
          --pickle_test ../Dataset_3/bcep/D_FSHH_ysctrnivlk_test.pickle \
          --model_path ./net/0622.pt
echo '-------test end---------'


echo '-------train---------'
python3 multiclass.py \
          --epoch 10 \
          --train 1 \
          --dim_in 12 \
          --label_size 6 \
          --pickle_train ../Dataset_3/bcep/D_FSHH_ysctrnivlkot_train.pickle \
          --pickle_valid ../Dataset_3/bcep/D_FSHH_ysctrnivlkot_valid.pickle \
          --pickle_test ../Dataset_3/bcep/D_FSHH_ysctrnivlkot_test.pickle \
          --model_path ./net/0622.pt
          #--csv_path ./log/0430_${ARG}.csv
echo '-------test ot---------'
python3 multiclass.py \
          --epoch 10 \
          --train 0 \
          --dim_in 12 \
          --label_size 6 \
          --pickle_train ../Dataset_3/bcep/D_FSHH_ysctrnivlkot_train.pickle \
          --pickle_valid ../Dataset_3/bcep/D_FSHH_ysctrnivlkot_valid.pickle \
          --pickle_test ../Dataset_3/bcep/D_FSHH_ysctrnivlkot_test.pickle \
          --model_path ./net/0622.pt
echo '-------test end---------'