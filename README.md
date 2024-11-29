# FastDiffusion.py
GenAI course project on fast diffusion models.
Primarily based upon
[iterative α-(de)blending](https://github.com/tchambon/IADB/tree/main)
and
[shortcut models](https://github.com/kvfrans/shortcut-models/tree/main).



## How to run

single gpu training
```bash
$ python -m fastdiff --case_dir unet_sm_test --mode 1 --train
```

multi-gpu training
```bash
$ torchrun --nproc-per-gpu gpu -m fastdiff --case_dir unet_sm_test --mode 1 --train # Shortcut model
$ torchrun --nproc-per-gpu gpu -m fastdiff --case_dir unet_fm_test --mode 0 --train # flow matching
```
