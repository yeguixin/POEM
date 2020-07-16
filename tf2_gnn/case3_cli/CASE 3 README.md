### CASE 3 README

本项目中提供了处理向量化任务的代码

#### 1.运行CASE 3

##### 配置项目中CASE 3相关文件

运行CASE 3相关代码之前的环境配置

```
# 进入case3对应的训练文件夹
cd case3_cli/ 

# 加载case3对应的训练文件, 无输出
source 00_set_caseFile.sh
```

##### 快速结果预览

通过加载已经训练好的模型，可以快速预览训练结果

```
# load模型, 输出准确率是0.67
source 02_rerun_to_get_result.sh model/bash_case3/*50_best.pkl
```

##### 训练模型

重新训练新模型

```
# 训练model
source 01_run_get_model.sh
```



#### 2.训练命令解析

```
python -u train.py case3 RGIN GraphBinaryClassification /dev/shm/zjq/case3_data/ast/ --save-dir trained_case3/ --load-trained-model trained_case3/*.pkl
# train.py训练主代码
# case3 选择case对应的函数
# RGIN 使用图网络
# GraphBinaryClassification 分类的结构
# /dev/shm/zjq/case3_data/ast/ --save-dir trained_case3/ 训练数据
# --save-dir trained_case3/ 模型保存路径
# --load-trained-model trained_case3/*.pkl load模型路径
```

