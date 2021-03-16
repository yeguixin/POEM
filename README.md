# POEM
POEM is a deep program structure modeling framework by leveraging a multi-relational graph neural network. It is built upon [tf2-gnn](https://github.com/microsoft/tf2-gnn), a graph neural network. POEM can capture the deep program semantic and structural features for code representation. We evaluated POEM by applying it to four representative tasks: [heterogeneous device mapping](#CS1), [GPU thread coarsening](#CS2), [loop vectorization](#CS3) and [code vulnerability detection](#CS4), and it gives the better performance than SOTA methods.
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


## Installation

POEM works on a machine running Linux with NVIDIA GPUs. The primary evaluations are conducted on a Linux server running Ubuntu 16.04 with a NVIDIA GTX Titan XP GPU.
Python 3.6 and Tensorflow 2.0.0 with CUDA are required to run POEM. Please refer to [this link](https://developer.nvidia.com/cuda-toolkit-archive) for installing CUDA Toolkits.
To get ready for running POEM, please additionally run the following commands:

```sh
$ conda create -n poem python=3.6
$ conda activate poem
$ git clone https://github.com/yeguixin/POEM.git
$ cd poem/src
$ pip install -e ./     /* installing multi-gnn  */
pip install -r req.txt  /* installing the requirements   */
```

## Dataset
The dataset used in our experiments are avaliable on our cloud disk by clicking the [archive link](https://pan.baidu.com/s/1QHyoCf0E7am1e2DfJTrv1w) or [the link](https://pan.baidu.com/s/1QHyoCf0E7am1e2DfJTrv1w) for our domestic users. The detailed usage of our dataset refers to [here](./data/README.md).


## Running

### Case Study 1: Heterogeneous Mapping <br id = "CS1">
In this task, we aim to build a predictive model to determine if the CPU or the GPU gives faster performance for a given OpenCL kernel. 

``` 
$ cd poem/src/tf2_gnn/case1_cli/
$ source 00_set_case1.sh              /* Configuration  */
$ source 01_run_get_model.sh          /* training the model */
$ source 02_rerun_to_get_result.sh    /* testing by using the trained model */
```

### Case Study 2: Thread Coarsening <br id = "CS2">
In this task, we aim to build a model to determine how many parallel threads should be merged together to achieve faster execution time.

``` 
$ cd poem/src/tf2_gnn/case2_cli/
$ python caseB-embedding-transfer.py
``` 

### Case Study 3: Loop Vectorization <br id = "CS3">
In this task, we aim to build a predictive model to determine the optimal vectorization factor (VF) and the interleaving factor (IF) for individual loops.

``` 
$ cd poem/src/tf2_gnn/case3_cli/
$ source 00_set_caseFile.sh       /* Configuration  */
$ source 01_run_get_model.sh      /* training the model */
$ source 02_rerun_to_get_result.sh    /* testing by using the trained model */
``` 

### Case Study 4: Vulnerability Detection   <br id = "CS4">
In this task, we build a model to detect if a given source code snippet contains one of the 2019 CWE top-25 most dangerous software errors at the function level.
The more details of this task refer to our [another paper](https://github.com/HuantWang/FUNDED_NISL) appeared in IEEE TIFS.


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
