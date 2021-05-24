#!/bin/bash
echo "source 01*.sh"


# 训练data/case1/new_*的10个例子, 将结果保存到当前路径下的res/result_new_*
for i in {0..9} ;do
    echo ${i}
    rm -rf res_$1_new_${i}/
    python -u train.py case1 RGIN GraphBinaryClassification ../../../data/case1/new_${i}/ast \
    --save-dir res/result_new_${i}/
done



# if [ $1 == amd ] ;then
#     echo "train amd ---"
#     for i in {0..9} ;do
#         echo ${i}
#         rm -rf result_$1_new_${i}/
#         python -u train.py case1 RGIN GraphBinaryClassification /dev/shm/zjq/case1_data/$1/new_${i}/ast --save-dir $2/result_$1_new_${i}/
#     done
# elif [ $1 == nvidia ] ;then
#     echo "train nvidia ---"
#     echo "train amd ---"
#     for i in {0..9} ;do
#         echo ${i}
#         rm -rf result_$1_new_${i}/
#         python -u train.py case1 RGIN GraphBinaryClassification /dev/shm/zjq/case1_data/$1/new_${i}/ast --save-dir $2/result_$1_new_${i}/
#     done
# elif [ $1 == big ] ;then 
#     echo "train big ---"
#     python -u train.py case1 RGIN GraphBinaryClassification /dev/shm/zjq/case1_data/data_1w/erwei_1w/ast --save-dir $2/result_$1/
# else
#     echo "please input amd/nvidia/big"
# fi



