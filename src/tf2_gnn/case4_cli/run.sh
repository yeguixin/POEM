rm -rf trained_model*/

if [ $1 == 1 ] ;then
    echo "python -u train.py case1 RGIN GraphBinaryClassification /dev/shm/zjq/case1_data/amd/new_2/ast --save-dir trained_case1/"
    python -u train.py case1 RGIN GraphBinaryClassification /dev/shm/zjq/case1_data/amd/new_2/ast --save-dir trained_case1/
elif [ $1 == 2 ] ;then
    echo "run case2"
elif [ $1 == 3 ] ;then 
    echo "python -u train.py case3 RGIN GraphBinaryClassification /dev/shm/zjq/case3_data/ast/"
    python -u train.py case3 RGIN GraphBinaryClassification /dev/shm/zjq/case3_data/ast/
elif [ $1 == 4 ] ;then    
    echo "CUDA_VISIBLE_DEVICES=0 python3 -u train.py case4 RGIN GraphBinaryClassification /dev/shm/zjq/data/CWE-200"
    CUDA_VISIBLE_DEVICES=0 python3 -u train.py case4 RGIN GraphBinaryClassification /dev/shm/zjq/data/CWE-200
fi
