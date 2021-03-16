# POEM
POEM is a deep program structure modeling framework by leveraging the multi-relational graph neural network. It can capture the deep program semantic and structural features for code representation. We evaluated POEM by applying it to four representative tasks: heterogeneous device mapping, GPU thread coarsening, loop vectorization and code vul- nerability detection, and it gives the better performance than SOTA methods.
For more details, please refer to our [paper](https://dl.acm.org/doi/abs/10.1145/3410463.3414670), "Deep Program Structure Modeling Through Multi-Relational Graph-based Learning", which appeared in PACT 2020.

## Abstract
> Deep learning is emerging as a promising technique for building predictive models to support code-related tasks like performance
optimization and code vulnerability detection. One of the critical aspects of building a successful predictive model is having the right
representation to characterize the model input for the given task. Existing approaches in the area typically treat the program structure
as a sequential sequence but fail to capitalize on the rich semantics of data and control flow information, for which graphs are a proven
representation structure.

> We present POEM, a novel framework that
automatically learns useful code representations from graph-based program structures. At the core of POEM is a graph neural
network (GNN) that is specially designed for capturing the syntax and semantic information from the program abstract syntax tree and the
control and data flow graph. As a departure from existing GNN-based code modeling techniques, our network simultaneously learns over
multiple relations of a program graph. This capability enables the learning framework to distinguish and reason about the diverse code
relationships, be it a data or a control flow or any other relationships that may be important for the downstream processing task.

> We apply POEM to four representative tasks that require a strong ability to reason about the program structure: heterogeneous
device mapping, parallel thread coarsening, loop vectorization and code vulnerability detection. We evaluate POEM on programs
written in OpenCL, C, Java and Swift, and compare it against nine learning-based methods. Experimental results show that POEM
consistently outperforms all competing methods across evaluation settings.


## 软件架构
软件以tf-gnn为基, 


## 安装教程

POEM可在运行带有NVIDIA图形卡的Linux的计算机上工作。它已在运行带有GTX Titan XP GPU的Ubuntu 20.04的计算机上进行了测试。  
- 具体安装这里利用conda创建python运行环境, 具体命令如下

```sh
conda create -n poem python=3.6
conda activate poem
git clone POEM的GitHub链接
cd poem/src
pip install -e ./ # 安装当前文件中的tf2-gnn到环境中
pip install -r req.txt # 加载其他相关包
```

## 数据集
数据集已经上传到网络云盘([点击这里进行下载, 提取码ntt6](https://pan.baidu.com/s/1QHyoCf0E7am1e2DfJTrv1w))  
下载后放置 `poem/data/data.zip`  
使用命令进行解压 `cd poem/data/ && unzip data.zip`
> `case1` 数据说明  
- case1的数据集中有三个数据集, amd/nvidia/1w zip文件分别是小数据集在amd/nvidia平台标记的数据集, 1w是大数据集在nvidia平台上标记的数据集 
- 对应的三个csv文件分别是上述中, 对比实验和相关参数表

> `case2` 数据说明
- `caseb_128.npy`文件是处理打包后的数据集 
- csv文件是实验数据相关参数表

> `case3` 数据说明
- 三套gz文件, 分别表示辅助输入数据集, for循环次数数据集, 和cdfg边数据集

## 使用说明
本文包含四个case, 

> `case1` 使用说明
- 路径: `poem/src/tf2_gnn/case1_cli/`
- 配置case1代码: `source 00_set_case1.sh` 
- 训练模型: `source 01_run_get_model.sh`
- 重载模型: `source 02_rerun_to_get_result.sh`

> `case2` 使用说明
- 路径: `poem/src/tf2_gnn/case2_cli/`
- 训练和测试: `python caseB-embedding-transfer.py`


> `case3` 使用说明
- 路径: `poem/src/tf2_gnn/case3_cli/`
- 配置case1代码: `source 00_set_caseFile.sh` 
- 训练模型: `source 01_run_get_model.sh`
- 重载模型: `source 02_rerun_to_get_result.sh`

> `case4`  
- case4涉及到的代码漏洞检测已经在另一个GitHub进行开源
- 具体请跳转到[这里](https://github.com/HuantWang/FUNDED_NISL)


# Citation
```
@inproceedings{ye2020deep,
  title={Deep Program Structure Modeling Through Multi-Relational Graph-based Learning},
  author={Ye, Guixin and Tang, Zhanyong and Wang, Huanting and Fang, Dingyi and Fang, Jianbin and Huang, Songfang and Wang, Zheng},
  booktitle={Proceedings of the ACM International Conference on Parallel Architectures and Compilation Techniques},
  pages={111--123},
  year={2020}
}
```