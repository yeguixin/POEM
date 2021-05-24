#!/bin/bash
# .sh platform save path
echo "source 02_rerun_to_get_result.sh amd model/amd_10/"
python -u train.py case3 RGIN GraphBinaryClassification ../../../data/case3/ast/ \
--save-dir model/ --load-trained-model $1  --load-saved-model $1




