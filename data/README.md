# Dataset Usage
Place the downloaded the compressed dataset in `poem/data/` path;

Then decompressing using the command `cd poem/data/ && unzip data.zip`

# Specification

## Case1
There are three kinds of compressed files which are introduced as follows:
- **amd:** This file contains about 680 OpenCL kernels that labeled under AMD platform.
- **nvidia:** This file contains about 680 OpenCL kernels that labeled under NVIDIA platform.
- **1w:** This file contains the 10,000 OpenCL kernels that labeled under NVIDIA plarform.

Note that there are also three csv files. They respectively are the parameters of the comparison experiments by using the above three data sets.


## Case2

- **caseb_128.npy:** This file is the graph data of 17 OpenCL kernels.
- **csv file:** The parameters in the comparison experiments.


## Case3

- **gz files:** Three files are the data sets respectively corresponding to auxiliary input, loops, and cdfg edges.
