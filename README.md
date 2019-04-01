
# Discrete Rotation Equivariance for Point Cloud Recognition
**Discrete Rotation Equivariance for Point Cloud Recognition.** ICRA 2019, Montreal, Canada.
Jiaxin Li, Yingcai Bi, Gim Hee Lee, National University of Singapore



## Introduction
This paper and repository propose a rotation equivariant method to improve deep network's performance 
when the input point cloud is rotated in 2D or 3D.


## Installation
Requirements:
- Python 3
- [PyTorch 0.4.1 or above](http://pytorch.org/)
- [Faiss](https://github.com/facebookresearch/faiss)
- [visdom](https://github.com/facebookresearch/visdom)

Optional dependency:
- Faiss [GPU support](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md)

## Dataset
- ModelNet40 and MNIST
 Our dataset is provided by [SO-Net](https://github.com/lijx10/SO-Net):
  
  [Google Drive](https://drive.google.com/open?id=184MbflF_RbDX9MyML3hid7OxsYJ8oQQ7): MNIST, ModelNet40/10
 
## Usage
### Compile Cuda Operations
run `python3 setup.py install` in `models/index_mat_ext` to install the customized maxpooling layer.

### Configurations
To run these tasks, you may need to set the dataset type and path in `options.py`, by changing the default value of `--dataset`, `--dataroot`.

In training, you may want to configure the input rotation settings in `options.py`
- Input rotation augmentation mode: `rot_equivariant_mode`, supports `'2d' / '3d''` 
- Number of descrete rotation: `rot_equivariant_no`. Supports `{1, 4, 6, 9, 12, 24}` when `rot_equivariant_mode=2d`, 
`{1, 4, 12}` when `rot_equivariant_mode=3d`. 

### Visualization
We use visdom for visualization. Various loss values and the reconstructed point clouds (in auto-encoder) are plotted in real-time. Please start the visdom server before training, otherwise there will be warnings/errors, though the warnings/errors won't affect the training process.
```
python3 -m visdom.server
```
The visualization results can be viewed in browser with the address of:
```
http://localhost:8097
```


## License
This repository is released under MIT License (see LICENSE file for details).

