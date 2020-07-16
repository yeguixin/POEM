#!/bin/bash
# .sh platform save_path 注意这个savepath是trainFile的上一层
echo "source 01*.sh amd model/amd_10/"

# GraphBinaryClassification(分类)+数据路径+保存路径
python -u train.py case3 RGIN GraphBinaryClassification /dev/shm/zjq/case3_data/ast/ --save-dir model/



