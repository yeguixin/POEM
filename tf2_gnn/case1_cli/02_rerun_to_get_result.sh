#!/bin/bash
# .sh platform save path
echo "source 02_rerun_to_get_result.sh amd model/amd_10/"
echo "start" > $1_result.log
if [ $1 == amd ] ; then
    for i in {0..9} ;do
        python -u train.py case1 RGIN GraphBinaryClassification /dev/shm/zjq/case1_data/$1/new_${i}/ast --save-dir swap/ --load-trained-model $2/result_$1_new_${i}/*.pkl >> $1_result.log
    done
elif [ $1 == nvidia ] ; then
    for i in {0..9} ;do
        python -u train.py case1 RGIN GraphBinaryClassification /dev/shm/zjq/case1_data/$1/new_${i}/ast --save-dir swap/ --load-trained-model $2/result_$1_new_${i}/*.pkl >> $1_result.log
    done
elif [ $1 == big ] ; then
    python -u train.py case1 RGIN GraphBinaryClassification /dev/shm/zjq/case1_data/data_1w/erwei_1w/ast --save-dir swap/ --load-trained-model $2/RGIN_GraphBi*.pkl >> $1_result.log
fi

python get_result_10.py $1_result.log $1 






