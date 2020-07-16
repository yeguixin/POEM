#!/bin/bash
# .sh platform save_path 注意这个savepath是trainFile的上一层
echo "source 01*.sh amd model/amd_10/"

if [ $1 == amd ] ;then
    echo "train amd ---"
    for i in {0..9} ;do
        echo ${i}
        rm -rf result_$1_new_${i}/
        python -u train.py RGIN GraphBinaryClassification /dev/shm/zjq/case1_data/$1/new_${i}/ast --save-dir $2/result_$1_new_${i}/
    done
elif [ $1 == nvidia ] ;then
    echo "train nvidia ---"
    echo "train amd ---"
    for i in {0..9} ;do
        echo ${i}
        rm -rf result_$1_new_${i}/
        python -u train.py RGIN GraphBinaryClassification /dev/shm/zjq/case1_data/$1/new_${i}/ast --save-dir $2/result_$1_new_${i}/
    done
elif [ $1 == big ] ;then 
    echo "train big ---"
    python -u train.py case1 RGIN GraphBinaryClassification /dev/shm/zjq/case1_data/data_1w/erwei_1w/ast --save-dir $2/result_$1/
else
    echo "please input amd/nvidia/big"
fi



