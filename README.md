# Multiple Semantic Relationships-Aware Transformer for Heterogeneous Graph Embedding (MSHGT)

The camera-ready paper for AAAI 23 can be found at: [http://arxiv.org/abs/2207.02547](http://arxiv.org/abs/2207.02547)

## Requirements

#### 1. Neural network libraries for GNNs

* [pytorch](https://pytorch.org/get-started/locally/)
* [pytorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

Please check your cuda version first and install the above libraries matching your cuda. If possible, we recommend to install the latest versions of these libraries.

#### 2. Other dependencies

Install other requirements:

```setup
pip install -r requirements.txt
git clone https://github.com/Yangxc13/sparse_tools.git --depth=1
cd sparse_tools
python setup.py develop
cd ..
```
