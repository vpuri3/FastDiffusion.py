# FastDiffusion.py

Generating samples from diffusion models involves many denoising steps leading to large latency.
In this project, we build upon two recent works to produce diffusion models that produce accurate samples with only a few denoising steps:
first, [shortcut diffusion models](https://github.com/kvfrans/shortcut-models/tree/main) [1] allow for taking arbitrary-sized denoising steps in the context of flow-matching;
second, [2] proposed the [trig flow](https://arxiv.org/pdf/2410.11081) formulation of the diffusion process in relation to [consistency models](https://arxiv.org/pdf/2303.01469).

Our proposed method redefines shortcut diffusion models in the context of trig flow by integrating a trigonometric noise schedule.
Specifically, our key contributions are
- Formulation of shortcut models in context of trig flow
- Development of a training objective and first-order sampler for trig flow
- Development of a training objective for trig flow for shortcut models

Our preliminary results are detailed in a [technical report](reporting/report.pdf) and summarized in a [poster](reporting/poster.pdf) rendered below.

![image](https://github.com/user-attachments/assets/fe897324-2cf4-485c-a23e-78d3f437d065)

[1] Liu X et al. ICLR. 2023.

[2] Lu C et al. arXiv:2410.1108. 2024.

## How to run

On a single GPU:
```bash
$ python -m fastdiff --case_dir flow-matching --mode 0 --train
$ python -m fastdiff --case_dir trig-flow --mode 0 --trig --train

$ python -m fastdiff --case_dir shortcut --mode 1 --train
$ python -m fastdiff --case_dir trig-shortcut --mode 1 --trig --train
```

Multi-GPU:
```bash
$ torchrun --nproc-per-gpu gpu -m fastdiff [ARGS]
```

## Acknowledgments
This was a course project for CMU 10-623 Generative AI course with teammates [Nihali Shetty](https://github.com/NihaliShetty) and [Nikhitha Beedala](https://github.com/binks07) during the Fall 2024 semester.

## Citation

```bibtex
@software{puri2024fastdiffusion,
  author    = {Puri, Vedant},
  title     = {{FastDiffusion.py: Accurate sampling with few step diffusion}},
  month     = dec,
  year      = 2024,
  publisher = {GitHub},
  url       = {https://github.com/vpuri3/FastDiffusion.py}
}
```
