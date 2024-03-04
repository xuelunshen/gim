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

- [x] Inference code
- [ ] Training code

> 很感谢大家对 GIM 的关注，我最近正在赶论文的 DDL，在论文完成之后，我会继续 GIM 的开源，尽快将 GIM 的代码分享给大家，感谢大家的耐心等待。预计三月下旬会继续代码的公开。

## 🤗 在线体验

去 [Huggingface](https://huggingface.co/spaces/xuelunshen/gim-online) 在线快速体验我们模型的效果

## ⚙️ 运行环境

我的代码运行环境是:
- `GeForce RTX 3090`
- `Ubuntu 20.04.3`
- `Python (3.8.10)`
- `Pytorch 1.10.2 (py3.8_cuda11.3_cudnn8.2.0_0)`

具体的环境请在安装 `anaconda` 之后运行下面的命令进行安装
```bash
conda env create -f environment.yml
```
如果上面的命令不能直接一键安装环境，请参考`environment.txt`中每个包的版本进行环境的安装。

## 🔨 使用

克隆我们的仓库, 然后运行下面的命令
```bash
python demo.py
```

代码会将 `assets/demo` 中的 `a.png` 和 `b.png` 进行匹配</br>
输出 `a_b_match.png` 和 `a_b_warp.png`

<details>
<summary>
	点击这里查看
	<code>a.png</code>
	和
	<code>b.png</code>.
</summary>
<p float="left">
  <img src="assets/demo/a.png" width="25%" />
  <img src="assets/demo/b.png" width="25%" /> 
</p>
</details>



<details>
<summary>
	点击这里查看
	<code>a_b_match.png</code>.
</summary>
<p align="left">
	<img src="assets/demo/_a_b_match.png" width="50%">
</p>
<p><code>a_b_match.png</code> 是两张图像匹配的可视化</p>
</details>

<details>
<summary>
	点击这里查看
	<code>a_b_warp.png</code>.
</summary>
<p align="left">
	<img src="assets/demo/_a_b_warp.png" width="50%">
</p>
<p><code>a_b_warp.png</code> 是将图像b用 homography 投影到图像a的效果</p>
</details>

## 📌 引用

如果我们的代码对你的研究有帮助, 请给我们的论文一个引用吧 ❤️ 多谢啦.

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
