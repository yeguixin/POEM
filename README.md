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

# Installation


# Dataset

# Usage




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
