## :book: Graph Transformer-based Heterogeneous Graph Neural Networks Enhanced by Multiple Meta-path Adjacency Matrices Decomposition
<image src="demo.jpg" width="100%">

# Introduction
This is a release of the code of our paper **_Graph Transformer-based Heterogeneous Graph Neural Networks Enhanced by Multiple Meta-path Adjacency Matrices Decomposition_**.

Authors:
XXXXX, 
XXXXX, 
XXXXX, 
XXXXX

[[code]](https://github.com/Lsb0113/MSHGT)

# Dependencies
```bash
conda create -n MSHGT python=3.8
conda activate MSHGT
pip install -r requirement.txt
pip install torch==1.12.1+cu113
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.12.1+cu113.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.12.1+cu113.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.12.1+cu113.html
pip install torch-geometric
```

# Prepare the data
You can use download IMDB, DBLP, LastFM by using dataset class in torch_geometric.datasets. For Pubmed, ACM and Yelp, please them from [the source of HGB benchmark] (https://cloud.tsinghua.edu.cn/d/fc10cb35d19047a88cb1/?p=NC)
```
# Run Code
Before running the main file, you can modify some of the experimental settings to suit your needs.
```bash
python main.py




