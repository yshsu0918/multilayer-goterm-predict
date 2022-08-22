echo '-------ONLYBW---------'
echo '-------ONLYBW---------'
echo '-------ONLYBW---------'

echo '-------test---------'
python3 multiclass.py \
          --epoch 10 \
          --train 0 \
          --dim_in 4 \
          --label_size 5 \
          --pickle_train ../Dataset/onlybw/D_FSHH_ysctrnivlk_train.pickle \
          --pickle_valid ../Dataset/onlybw/D_FSHH_ysctrnivlk_valid.pickle \
          --pickle_test ../Dataset/onlybw/D_FSHH_ysctrnivlk_test.pickle \
          --eng_abbrs ys,ct,rn,iv,lk \
          --model_path ./net/0817_onlybw.pt
echo '-------test end---------'




echo '-------BC---------'
echo '-------BC---------'
echo '-------BC---------'

echo '-------test---------'
python3 multiclass.py \
          --epoch 10 \
          --train 0 \
          --dim_in 8 \
          --label_size 5 \
          --pickle_train ../Dataset/bc/D_FSHH_ysctrnivlk_train.pickle \
          --pickle_valid ../Dataset/bc/D_FSHH_ysctrnivlk_valid.pickle \
          --pickle_test ../Dataset/bc/D_FSHH_ysctrnivlk_test.pickle \
          --eng_abbrs ys,ct,rn,iv,lk \
          --model_path ./net/0817_onlybc.pt
echo '-------test end---------'




echo '-------BCE---------'
echo '-------BCE---------'
echo '-------BCE---------'


echo '-------test---------'
python3 multiclass.py \
          --epoch 10 \
          --train 0 \
          --dim_in 10 \
          --label_size 5 \
          --pickle_train ../Dataset/bce/D_FSHH_ysctrnivlk_train.pickle \
          --pickle_valid ../Dataset/bce/D_FSHH_ysctrnivlk_valid.pickle \
          --pickle_test ../Dataset/bce/D_FSHH_ysctrnivlk_test.pickle \
          --eng_abbrs ys,ct,rn,iv,lk \
          --model_path ./net/0817_onlybce.pt
echo '-------test end---------'




echo '-------BCEP---------'
echo '-------BCEP---------'
echo '-------BCEP---------'

echo '-------test---------'
python3 multiclass.py \
          --epoch 10 \
          --train 0 \
          --dim_in 12 \
          --label_size 5 \
          --pickle_train ../Dataset/bcep/D_FSHH_ysctrnivlk_train.pickle \
          --pickle_valid ../Dataset/bcep/D_FSHH_ysctrnivlk_valid.pickle \
          --pickle_test ../Dataset/bcep/D_FSHH_ysctrnivlk_test.pickle \
          --eng_abbrs ys,ct,rn,iv,lk \
          --model_path ./net/0817_onlybcep.pt
echo '-------test end---------'

