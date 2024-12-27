<p align="center">
  <a href="README.md"><img src="https://img.shields.io/badge/English-white" alt='English'></a>
  <a href="README.zh-CN-simplified.md"><img src="https://img.shields.io/badge/%E4%B8%AD%E6%96%87-white" alt='Chinese'></a>
</p>

<h2 align="center">GIM: Learning Generalizable Image Matcher From Internet Videos</h2>


<div align="center">
	<a href="https://www.youtube.com/embed/FU_MJLD8LeY">
		<img src="assets/demo/video.png" width="50%" alt="Overview Video">
	</a>
</div>
<p></p>

<div align="center">

<!-- <a href="https://iclr.cc/Conferences/2024"><img src="https://img.shields.io/badge/%F0%9F%8C%9F_ICLR'2024_Spotlight-37414c" alt='ICLR 2024 Spotlight'></a> -->
<a href="https://xuelunshen.com/gim"><img src="https://img.shields.io/badge/Project_Page-3A464E?logo=gumtree" alt='Project Page'></a>
<a href="https://arxiv.org/abs/2402.11095"><img src="https://img.shields.io/badge/arXiv-2402.11095-b31b1b?logo=arxiv" alt='arxiv'></a>
<a href="https://huggingface.co/spaces/xuelunshen/gim-online"><img src="https://img.shields.io/badge/%F0%9F%A4%97_Hugging_Face-Space-F0CD4B?labelColor=666EEE" alt='HuggingFace Space'></a>
<a href="https://www.youtube.com/watch?v=FU_MJLD8LeY"><img src="https://img.shields.io/badge/Video-E33122?logo=Youtube" alt='Overview Video'></a>
<a href="https://community.intel.com/t5/Blogs/Tech-Innovation/Artificial-Intelligence-AI/Intel-Labs-Research-Work-Receives-Spotlight-Award-at-Top-AI/post/1575985"><img src="https://img.shields.io/badge/Blog-0071C5?logo=googledocs&logoColor=white" alt='Blog'></a>
<a href="https://zhuanlan.zhihu.com/p/711361901"><img src="https://img.shields.io/badge/Zhihu-1767F5?logo=zhihu&logoColor=white" alt='Blog'></a>
![GitHub Repo stars](https://img.shields.io/github/stars/xuelunshen/gim?style=social)

<!-- <a href="https://xuelunshen.com/gim"><img src="https://img.shields.io/badge/📊_Zero--shot_Image_Matching_Evaluation Benchmark-75BC66" alt='Zero-shot Evaluation Benchmark'></a> -->
<!-- <a href="https://xuelunshen.com/gim"><img src="https://img.shields.io/badge/Source_Code-black?logo=Github" alt='Github Source Code'></a> -->

<a href="https://en.xmu.edu.cn"><img src="https://img.shields.io/badge/XMU-183F9D?logo=Google%20Scholar&logoColor=white" alt='Intel'></a>
<a href="https://www.intel.com"><img src="https://img.shields.io/badge/Labs-0071C5?logo=intel" alt='Intel'></a>
<a href="https://www.dji.com"><img src="https://img.shields.io/badge/DJI-131313?logo=DJI" alt='Intel'></a>

</div>

## ✅ 待办清单

- [x] **ZEB**: **Z**ero-shot **E**valuation **B**enchmark
- [x] 视频处理代码
- [x] 三维重建
- [ ] 模型
  - [ ] gim_roma
  - [x] gim_dkm
  - [x] gim_loftr
  - [x] gim_lightglue
- [x] 训练代码

> 剩余的开源工作我们还在抓紧进行, 感谢大家的关注.

## 🤗 在线体验

去 [Huggingface](https://huggingface.co/spaces/xuelunshen/gim-online) 在线快速体验我们模型的效果

## ⚙️ 运行环境

我在新服务器上是使用下面的命令进行运行环境的安装.

<p></p>
<details>
<summary><b>[ 点击查看运行命令 ]</b></summary>

```bash
conda create -n gim python=3.9
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -c conda-forge
conda install xformers -c xformers
pip install albumentations==1.0.1 --no-binary=imgaug,albumentations
pip install colour-demosaicing==0.2.2
pip install pytorch-lightning==1.5.10
pip install opencv-python==4.5.3.56
pip install imagesize==1.2.0
pip install kornia==0.6.10
pip install einops==0.3.0
pip install loguru==0.5.3
pip install joblib==1.0.1
pip install yacs==0.1.8
pip install h5py==3.1.0
pip install matplotlib
pip install omegaconf
pip install triton
```

</details>
<p></p>

## 🔨 如何使用 `GIM` 系列的匹配网络

1. 克隆本仓库

```bash
git clone https://github.com/xuelunshen/gim.git
cd gim
```

2. 从 [Google Drive](https://drive.google.com/file/d/1gk97V4IROnR1Nprq10W9NCFUv2mxXR_-/view?usp=sharing) 或者 [OneDrive](https://stuxmueducn-my.sharepoint.com/:u:/g/personal/xuelun_stu_xmu_edu_cn/EdJOibZ8VABOoKoyOHWo8ZEBHd_MyHbSvhRyT_o40SIPGA?e=GCjGZE) 下载 `gim_dkm` 的模型参数

3. 将模型参数放在文件夹 `weights` 里面

4. 运行下面的命令

<p></p>
<details>
<summary><b>[ 点击查看运行命令 ]</b></summary>

```bash
python demo.py --model gim_dkm
# or
python demo.py --model gim_loftr
# or
python demo.py --model gim_lightglue
```

</details>
<p></p>


5. 代码会将 `assets/demo` 中的 `a1.png` 和 `a2.png` 进行匹配,</br>并且输出 `a1_a2_match.png` 和 `a1_a2_warp.png`

<details>
<summary>
<b>
	[ 点击这里查看
	<code>a1.png</code>
	和
	<code>a2.png</code> ]
</b>
</summary>
<p float="left">
  <img src="assets/demo/a1.png" width="25%" />
  <img src="assets/demo/a2.png" width="25%" /> 
</p>
</details>



<details>
<summary>
<b>
	[ 点击这里查看
	<code>a1_a2_match.png</code> ]
</b>
</summary>
<p align="left">
	<img src="assets/demo/_a1_a2_match.png" width="50%">
</p>
<p><code>a1_a2_match.png</code> 是两张图像匹配的可视化</p>
</details>

<details>
<summary>
<b>
	[ 点击这里查看
	<code>a1_a2_warp.png</code> ]
</b>
</summary>
<p align="left">
	<img src="assets/demo/_a1_a2_warp.png" width="50%">
</p>
<p><code>a1_a2_warp.png</code> 是将<code>图像a2</code>用 homography 投影到<code>图像a1</code>的效果</p>
</details>

<p></p>
还有更多图像在文件夹 `assets/demo` 中, 大家都可以尝试拿来匹配看看.
<p></p>

<details>
<summary>
<b>
	[ 点击这里查看更多图像 ]
</b>
</summary>
<p float="left">
  <img src="assets/demo/b1.png" width="15%" />
  <img src="assets/demo/b2.png" width="15%" /> 
  <img src="assets/demo/c1.png" width="15%" />
  <img src="assets/demo/c2.png" width="15%" /> 
  <img src="assets/demo/d1.png" width="15%" />
  <img src="assets/demo/d2.png" width="15%" /> 
</p>
</details>

## 🎞️ 视频处理
### 不需要三维重建即可得到视频图像帧之间可靠的像素对应关系
> 因为一些原因, 我们不能提供具体使用了哪些 Youtube 视频进行训练, 我可以告诉大家的是, 用关键词 `walk in` 或者 `walk through` 去 YouTube 搜索相关视频. 用来处理的视频需要是拍摄后没有经过任何处理的. 不要有剪辑, 不要有转场, 不要有特效等等. 下面我介绍一下整个流程.

> 准备工作: 从 [Google Drive](https://drive.google.com/file/d/1YswCj58VuVhqEpMKQ_k0QJb3_mMdpF8M/view?usp=sharing) 或者 [OneDrive](https://stuxmueducn-my.sharepoint.com/:u:/g/personal/xuelun_stu_xmu_edu_cn/EUR_XMay5b5FtWelmqXiLi4Bcnv4G1w5b2aYjhqS-Ds_ow) 下载来自 [semantic-segmentation-pytorch](https://github.com/CSAILVision/semantic-segmentation-pytorch) 的模型参数(`decoder_epoch_20.pth`), 将模型参数放在文件夹 `weights` 里面.

1. 将你要处理的 YouTube 视频的 id 粘贴到 `video_list.txt` 文件中. 比如视频 `https://www.youtube.com/watch?v=Od-rKbC30TM` 的 id 就是 `Od-rKbC30TM`. 现在 video_list.txt 文件内已经粘贴了这个示例视频. 你现在可以先什么都不用做, 直接进入第二步.
2. 用命令 `chmod +x process_videos.sh` 赋予 `process_videos.sh` 文件执行权限
3. 用命令 `./process_videos.sh video_list.txt` 运行视频处理代码
4. 用命令 `python -m datasets.walk.propagate video_list.txt` 运行匹配传递代码
5. 用命令 `python -m datasets.walk.walk video_list.txt` 运行可视化代码

> 处理结果和中间文件位于 `data/ZeroMatch` 文件夹内, 可视化结果在 `dump/walk` 文件夹内. 不出意外你可以看到类似下方图片的处理结果(点击展开图像).

<details>
<summary>
<b>
	[ 点击这里查看视频处理可视化结果 ]
</b>
</summary>
<p align="left">
	<img src="assets/demo/example.png" width="50%">
</p>
</details>

<details>
<summary>
<b>
	[ ⚠️ 如果你遇到 torchvision 的 VideoReader 报错, 请点击展开 ]
</b>
</summary>
新建一个 conda 环境并且参考下面的内容安装依赖, 再去运行视频处理代码.

```bash
conda create -n gim-video python=3.8.10
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install albumentations==1.0.1 --no-binary=imgaug,albumentations
pip install pytorch-lightning==1.5.10
pip install opencv-python==4.5.3.56
pip install imagesize==1.2.0
pip install kornia==0.6.10
pip install einops==0.3.0
pip install loguru==0.5.3
pip install joblib==1.0.1
pip install yacs==0.1.8
pip install h5py==3.1.0
```

</details>

## 🏋️ 训练网络
> 处理完视频之后就是训练网络, 训练 `gim-loftr` 的代码在仓库分支 `train-gim-loftr` 中. 训练 `gim-dkm` 的代码和训练 `gim-lightglue` 的代码稍后会开源. 不过相比于 `loftr`, 适配 gim 的视频数据到 `dkm` 和 `lightglue` 的架构其实简单的多, 所以我们先公布 `gim-loftr` 的训练代码.

1. 用命令 `git checkout train-gim-loftr` 切换到 `train-gim-loftr` 分支
2. 用下方命令运行训练代码

```bash
#! /bin/bash
GPUS=8
NNODES=5
GITID=$(git rev-parse --short=8 HEAD)
MODELID=$(cat /dev/urandom | tr -dc 'a-z0-9' | fold -w 8 | head -n 1)
python -m torch.distributed.launch --nproc_per_node=gpu --nnodes=$WORLD_SIZE --node_rank $RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT --use_env train.py --num_nodes $NNODES --gpus $GPUS --max_epochs 10 --maxlen 938240 938240 938240 --lr 0.001 --min_lr 0.00005 --git $GITID --wid $MODELID --resample --img_size 840 --batch_size 1 --valid_batch_size 2
```

我们是在 5 个 A100 节点上进行 `gim-loftr` 的训练, 每个节点 8 张 80 GB 的显卡. 其中 `WORLD_SIZE`, `RANK`, `MASTER_ADDR`, `MASTER_PORT` 是分布式训练的参数, 应该可以自动从集群运行环境中获取. 如果你用的是单机单卡或者单机多卡训练, 那么用下面的命令运行训练代码即可.

```bash
python train.py --num_nodes 1 --gpus $GPUS --max_epochs 10 --maxlen 938240 938240 938240 --lr 0.001 --min_lr 0.00005 --git $GITID --wid $MODELID --resample --img_size 840 --batch_size 1 --valid_batch_size 2
```

## 🕋 三维重建

本仓库三维重建的代码是基于 [hloc](https://github.com/cvg/Hierarchical-Localization) 实现.

首先, 按照 hloc 的 README 安装 [colmap](https://colmap.github.io/) 和 [pycolmap](https://github.com/colmap/pycolmap).

然后, 从 [Google Drive](https://drive.google.com/file/d/1YswCj58VuVhqEpMKQ_k0QJb3_mMdpF8M/view?usp=sharing) 或者 [OneDrive](https://stuxmueducn-my.sharepoint.com/:u:/g/personal/xuelun_stu_xmu_edu_cn/EUR_XMay5b5FtWelmqXiLi4Bcnv4G1w5b2aYjhqS-Ds_ow) 下载来自 [semantic-segmentation-pytorch](https://github.com/CSAILVision/semantic-segmentation-pytorch) 的模型参数(`decoder_epoch_20.pth`), 将模型参数放在文件夹 `weights` 里面.

接着, 创建一些文件夹, 假如想要对房间做三维重建, 运行下面的命令:

```bash
mkdir -p inputs/room/images
```

然后, 将要进行三维重建的若干张房间图片放到 `images` 文件夹内.

最后运行下面的命令即可进行三维重建:

```bash
sh reconstruction.sh room gim_dkm
# or
sh reconstruction.sh room gim_lightglue
```

> Tips:\
> 目前三维重建的代码默认会将所有图片两两配对, 然后进行图像匹配和重建\
> 为了更好的重建结果, 建议根据实际情况修改代码, 对配对图片进行调整.

## 📊 ZEB: Zero-shot Evaluation Benchmark

1. 创建一个名为 **`zeb`** 的文件夹
2. 从[这个网址](https://stuxmueducn-my.sharepoint.com/:f:/g/personal/lizijun_stu_xmu_edu_cn/EmHLjQpbpDRKmiED88rxGl4BFIkSp7vAzXifwXtvVbQA9w?e=ey8WVk)下载 **ZEB** 测试数据的 zip 压缩包, 将其放在刚才创建的 **`zeb`** 文件夹内, 并且解压 zip 压缩包.
3. 运行下面命令开始测试

<p></p>
<details>
<summary><b>[ 点击查看运行命令 ]</b></summary>

下面的数字 **1** 代表你要使用的 gpu 数量,如果你想用 **2** 块gpu, 则将数字 **1** 改为 **2**.

```bash
sh TEST_GIM_DKM.sh 1
# or
sh TEST_GIM_LOFTR.sh 1
# or
sh TEST_GIM_LIGHTGLUE.sh 1
# or
sh TEST_ROOT_SIFT.sh 1
```
</details>
<p></p>

4. 运行命令 `python check.py` 来检查是否输出全是 `"Good"`.
5. 运行命令 `python analysis.py --dir dump/zeb --wid gim_dkm --version 100h --verbose` 来取得 **ZEB** 测试结果.
6. 将 **ZEB** 测试结果粘贴到名为 `zeb.xlsx` 的 Excel 文件中.

<p></p>
<details>
<summary><b><font color="red">[ 点击显示 📊 ZEB 测试结果 ]</font></b></summary>

> 该表格的数据来自论文提出的 **ZEB**: <u>Zero-shot Evaluation Benchmark for Image Matching</u>, 该 benchmark 由 12 个涵盖各种场景、天气和相机模型的公开数据集组成, 对应了表格中从 GL3 开始的 12 列测试序列.

|      | <div align="left">方法</div>                                 | <div align="left">平均<br />AUC@5°<br />(%) ↑</div> | GL3      | BLE      | ETI      | ETO      | KIT      | WEA      | SEA      | NIG      | MUL      | SCE      | ICL      | GTA      |
| ---- | ------------------------------------------------------------ | --------------------------------------------------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
|      |                                                              | 传统算法                                            |          |          |          |          |          |          |          |          |          |          |          |          |
|      | RootSIFT                                                     | 31.8                                                | 43.5     | 33.6     | 49.9     | 48.7     | 35.2     | 21.4     | 44.1     | 14.7     | 33.4     | 7.6      | 14.8     | 35.1     |
|      |                                                              | 稀疏匹配                                            |          |          |          |          |          |          |          |          |          |          |          |          |
|      | [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork) (in) | 21.6                                                | 19.2     | 16.0     | 38.2     | 37.7     | 22.0     | 20.8     | 40.8     | 13.7     | 21.4     | 0.8      | 9.6      | 18.8     |
|      | SuperGlue (out)                                              | 31.2                                                | 29.7     | 24.2     | 52.3     | 59.3     | 28.0     | 28.4     | 48.0     | 20.9     | 33.4     | 4.5      | 16.6     | 29.3     |
|      | **GIM_SuperGlue**<br />(50h)                                 | 34.3                                                | 43.2     | 34.2     | 58.7     | 61.0     | 29.0     | 28.3     | 48.4     | 18.8     | 34.8     | 2.8      | 15.4     | 36.5     |
|      | [LightGlue](https://github.com/cvg/LightGlue)                | 31.7                                                | 28.9     | 23.9     | 51.6     | 56.3     | 32.1     | 29.5     | 48.9     | 22.2     | 37.4     | 3.0      | 16.2     | 30.4     |
| ✅    | **GIM_LightGlue**<br />(100h)                                | **38.3**                                            | **46.6** | **38.1** | **61.7** | **62.9** | **34.9** | **31.2** | **50.6** | **22.6** | **41.8** | **6.9**  | **19.0** | **43.4** |
|      |                                                              | 半密集匹配                                          |          |          |          |          |          |          |          |          |          |          |          |          |
|      | [LoFTR](https://github.com/zju3dv/LoFTR) (in)                | 10.7                                                | 5.6      | 5.1      | 11.8     | 7.5      | 17.2     | 6.4      | 9.7      | 3.5      | 22.4     | 1.3      | 14.9     | 23.4     |
|      | LoFTR (out)                                                  | 33.1                                                | 29.3     | 22.5     | 51.1     | 60.1     | **36.1** | **29.7** | **48.6** | **19.4** | 37.0     | **13.1** | 20.5     | 30.3     |
| ✅    | **GIM_LoFTR**<br />(50h)                                     | **39.1**                                            | **50.6** | **43.9** | **62.6** | **61.6** | 35.9     | 26.8     | 47.5     | 17.6     | **41.4** | 10.2     | **25.6** | **45.0** |
|      | **GIM_LoFTR**<br />(100h)                                    | ToDO                                                |          |          |          |          |          |          |          |          |          |          |          |          |
|      |                                                              | 密集匹配                                            |          |          |          |          |          |          |          |          |          |          |          |          |
|      | [DKM](https://github.com/Parskatt/DKM) (in)                  | 46.2                                                | 44.4     | 37.0     | 65.7     | 73.3     | 40.2     | 32.8     | 51.0     | 23.1     | 54.7     | 33.0     | **43.6** | 55.7     |
|      | DKM (out)                                                    | 45.8                                                | 45.7     | 37.0     | 66.8     | 75.8     | 41.7     | 33.5     | 51.4     | 22.9     | 56.3     | 27.3     | 37.8     | 52.9     |
|      | **GIM_DKM**<br />(50h)                                       | 49.4                                                | 58.3     | 47.8     | 72.7     | 74.5     | 42.1     | **34.6** | 52.0     | **25.1** | 53.7     | 32.3     | 38.8     | 60.6     |
| ✅    | **GIM_DKM**<br />(100h)                                      | **51.2**                                            | **63.3** | **53.0** | **73.9** | 76.7     | **43.4** | **34.6** | **52.5** | 24.5     | 56.6     | 32.2     | 42.5     | **61.6** |
|      | [RoMa](https://github.com/Parskatt/RoMa) (in)                | 46.7                                                | 46.0     | 39.3     | 68.8     | 77.2     | 36.5     | 31.1     | 50.4     | 20.8     | 57.8     | **33.8** | 41.7     | 57.6     |
|      | RoMa (out)                                                   | 48.8                                                | 48.3     | 40.6     | 73.6     | **79.8** | 39.9     | 34.4     | 51.4     | 24.2     | **59.9** | 33.7     | 41.3     | 59.2     |
|      | **GIM_RoMa**                                                 | ToDO                                                |          |          |          |          |          |          |          |          |          |          |          |          |

</details>
<p></p>

## 🖼️ 海报

<div align="center">
	<a href="https://raw.githubusercontent.com/xuelunshen/gim/main/assets/demo/poster.png">
		<img src="assets/demo/poster.png" width="50%" alt="Overview Video">
	</a>
</div>

## 📌 引用

如果我们的代码对你的研究有帮助, 请给我们的论文一个引用 ❤️ 并给 gim 的仓库点个小星星 ⭐️ 吧, 多谢啦～

```bibtex
@inproceedings{
xuelun2024gim,
title={GIM: Learning Generalizable Image Matcher From Internet Videos},
author={Xuelun Shen and Zhipeng Cai and Wei Yin and Matthias Müller and Zijun Li and Kaixuan Wang and Xiaozhi Chen and Cheng Wang},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024}
}
```

## License

This repository is under the MIT License. This content/model is provided here for research purposes only. Any use beyond this is your sole responsibility and subject to your securing the necessary rights for your purpose. 
