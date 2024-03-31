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

<a href="https://iclr.cc/Conferences/2024"><img src="https://img.shields.io/badge/%F0%9F%8C%9F_ICLR'2024_Spotlight-37414c" alt='ICLR 2024 Spotlight'></a>
<a href="https://xuelunshen.com/gim"><img src="https://img.shields.io/badge/Project_Page-3A464E?logo=gumtree" alt='Project Page'></a>
<a href="https://arxiv.org/abs/2402.11095"><img src="https://img.shields.io/badge/arXiv-2402.11095-b31b1b?logo=arxiv" alt='arxiv'></a>
<a href="https://huggingface.co/spaces/xuelunshen/gim-online"><img src="https://img.shields.io/badge/%F0%9F%A4%97_Hugging_Face-Space-F0CD4B?labelColor=666EEE" alt='HuggingFace Space'></a>
<a href="https://www.youtube.com/watch?v=FU_MJLD8LeY"><img src="https://img.shields.io/badge/Overview_Video-E33122?logo=Youtube" alt='Overview Video'></a>
![GitHub Repo stars](https://img.shields.io/github/stars/xuelunshen/gim?style=social)

<!-- <a href="https://xuelunshen.com/gim"><img src="https://img.shields.io/badge/📊_Zero--shot_Image_Matching_Evaluation Benchmark-75BC66" alt='Zero-shot Evaluation Benchmark'></a> -->
<!-- <a href="https://xuelunshen.com/gim"><img src="https://img.shields.io/badge/Source_Code-black?logo=Github" alt='Github Source Code'></a> -->

<a href="https://en.xmu.edu.cn"><img src="https://img.shields.io/badge/Xiamen_University-183F9D?logo=Google%20Scholar&logoColor=white" alt='Intel'></a>
<a href="https://www.intel.com"><img src="https://img.shields.io/badge/Labs-0071C5?logo=intel" alt='Intel'></a>
<a href="https://www.dji.com"><img src="https://img.shields.io/badge/DJI-131313?logo=DJI" alt='Intel'></a>

</div>

## ✅ TODO List

- [ ] Inference code
  - [x] gim_dkm
  - [ ] gim_loftr
  - [x] gim_lightglue
- [ ] Training code

> We are actively continuing with the remaining open-source work and appreciate everyone's attention.

## 🤗 Online demo

Go to [Huggingface](https://huggingface.co/spaces/xuelunshen/gim-online) to quickly try our model online.

## ⚙️ Environment

I set up the running environment on a new machine using the commands listed below.
```bash
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

## 🔨 Usage

Clone the repository

```bash
git clone https://github.com/xuelunshen/gim.git
cd gim
mkdir weights
```

Download `gim_dkm` model weight from [Google Drive](https://drive.google.com/file/d/1gk97V4IROnR1Nprq10W9NCFUv2mxXR_-/view?usp=sharing)

Put it on the folder `weights`

Run the following command
```bash
python demo.py --model gim_dkm
```
or
```bash
python demo.py --model gim_lightglue
```

The code will match `a.png` and `b.png` in the folder `assets/demo`</br>, and output `a_b_match.png` and `a_b_warp.png`.

<details>
<summary>
	Click to show
	<code>a.png</code>
	and
	<code>b.png</code>.
</summary>
<p float="left">
  <img src="assets/demo/a.png" width="25%" />
  <img src="assets/demo/b.png" width="25%" /> 
</p>
</details>



<details>
<summary>
	Click to show
	<code>a_b_match.png</code>.
</summary>
<p align="left">
	<img src="assets/demo/_a_b_match.png" width="50%">
</p>
<p><code>a_b_match.png</code> is a visualization of the match between the two images</p>
</details>

<details>
<summary>
	Click to show
	<code>a_b_warp.png</code>.
</summary>
<p align="left">
	<img src="assets/demo/_a_b_warp.png" width="50%">
</p>
<p><code>a_b_warp.png</code> shows the effect of projecting `image b` onto `image a` using homography</p>
</details>

## 📌 Citation

If our code helps your research, please give a citation to our paper ❤️ Thank you very much.

```bibtex
@inproceedings{
xuelun2024gim,
title={GIM: Learning Generalizable Image Matcher From Internet Videos},
author={Xuelun Shen and Zhipeng Cai and Wei Yin and Matthias Müller and Zijun Li and Kaixuan Wang and Xiaozhi Chen and Cheng Wang},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024}
}
```

## 🌟 Star History

<a href="https://star-history.com/#xuelunshen/gim&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=xuelunshen/gim&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=xuelunshen/gim&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=xuelunshen/gim&type=Date" />
  </picture>
</a>

## License

This repository is under the MIT License. This content/model is provided here for research purposes only. Any use beyond this is your sole responsibility and subject to your securing the necessary rights for your purpose. 
