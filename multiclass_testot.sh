echo '-------ONLYBW---------'
echo '-------ONLYBW---------'
echo '-------ONLYBW---------'



echo '-------test ot---------'
python3 multiclass.py \
          --epoch 10 \
          --train 0 \
          --dim_in 4 \
          --label_size 6 \
          --pickle_train ../Dataset/onlybw/D_FSHH_ysctrnivlkot_train.pickle \
          --pickle_valid ../Dataset/onlybw/D_FSHH_ysctrnivlkot_valid.pickle \
          --pickle_test ../Dataset/onlybw/D_FSHH_ysctrnivlkot_test.pickle \
          --eng_abbrs ys,ct,rn,iv,lk,ot \
          --model_path ./net/0817_onlybw_ot.pt
echo '-------test end---------'


echo '-------BC---------'
echo '-------BC---------'
echo '-------BC---------'



echo '-------test ot---------'
python3 multiclass.py \
          --epoch 10 \
          --train 0 \
          --dim_in 8 \
          --label_size 6 \
          --pickle_train ../Dataset/bc/D_FSHH_ysctrnivlkot_train.pickle \
          --pickle_valid ../Dataset/bc/D_FSHH_ysctrnivlkot_valid.pickle \
          --pickle_test ../Dataset/bc/D_FSHH_ysctrnivlkot_test.pickle \
          --model_path ./net/0817_onlybc_ot.pt
echo '-------test end---------'


echo '-------BCE---------'
echo '-------BCE---------'
echo '-------BCE---------'



echo '-------test ot---------'
python3 multiclass.py \
          --epoch 10 \
          --train 0 \
          --dim_in 10 \
          --label_size 6 \
          --pickle_train ../Dataset/bce/D_FSHH_ysctrnivlkot_train.pickle \
          --pickle_valid ../Dataset/bce/D_FSHH_ysctrnivlkot_valid.pickle \
          --pickle_test ../Dataset/bce/D_FSHH_ysctrnivlkot_test.pickle \
          --model_path ./net/0817_onlybce_ot.pt
echo '-------test end---------'


echo '-------BCEP---------'
echo '-------BCEP---------'
echo '-------BCEP---------'



echo '-------test ot---------'
python3 multiclass.py \
          --epoch 10 \
          --train 0 \
          --dim_in 12 \
          --label_size 6 \
          --pickle_train ../Dataset/bcep/D_FSHH_ysctrnivlkot_train.pickle \
          --pickle_valid ../Dataset/bcep/D_FSHH_ysctrnivlkot_valid.pickle \
          --pickle_test ../Dataset/bcep/D_FSHH_ysctrnivlkot_test.pickle \
          --model_path ./net/0817_onlybcep_ot.pt
echo '-------test end---------'