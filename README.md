
# Colorization model - modularized

This folder contains a modularized and runnable extraction of the notebook [`pix2pix-image-colorization-with-conditional-wgan.ipynb`](https://www.kaggle.com/code/salimhammadi07/pix2pix-image-colorization-with-conditional-wgan/notebook).

## Contents

- `models.py` — generator (ResUnet) and critic (PatchGAN) + `build_models()` helper.
- `dataset.py` — `ImageColorizationDataset` returning (ab, L) tensors.
- `utils.py` — helpers for LAB <-> RGB conversion and saving images.
- `preprocess_images.py` — converts a folder of PNG/JPG into `l.npy` and `ab.npy` (see below).
- `finetune.py` — training script with WGAN-GP and optional R1 regularization. CLI supports resuming from weights.
- `inference.py` — inference CLI (single image or input folder -> output folder). Auto-finds latest weight in a directory.
- `requirements.txt` — packages used.

## Quick workflow

Example provided on [kaggle](https://www.kaggle.com/code/railsabirov/colorizationmodelfinetune)

1. Preprocess your RGB images (PNG/JPG) into numpy arrays:

```bash
python preprocess_images.py --input_dir /kaggle/input/my_images --out_dir /kaggle/working/prepared --size 224
```

This creates `l.npy` and `ab.npy` in the target folder. `l.npy` has shape (N, H, W, 1), `ab.npy` has shape (N, H, W, 2).

2. Train / finetune on Kaggle

```bash
python finetune.py \
	--ab_path /kaggle/working/prepared/ab.npy \
	--l_path /kaggle/working/prepared/l.npy \
	--epochs 50 --batch_size 8 --critic_steps 5 --lambda_gp 10 --out_dir ./checkpoints
```

If you want to resume from the latest checkpoint in a folder:

```bash
python finetune.py --ab_path ... --l_path ... --resume_gen ./checkpoints --resume_crit ./checkpoints
```

3. Inference

Single image:

```bash
python inference.py --input input_gray.png --output out.png --weights ./checkpoints
```

Batch (folder -> folder):

```bash
python inference.py --input /kaggle/input/my_images --output /kaggle/working/out_images --weights ./checkpoints
```

## Notes about weights and normalization

- The preprocessing and model code follow this convention:
	- L channel is scaled to [0,1] (when converting back to RGB we multiply by 100 to get the L* value for LAB).
	- ab channels are normalized to [0,1] with 0.5 corresponding to zero color offset. Concretely: ab_norm ~= (ab / 128) / 2 + 0.5 in the preprocessing step. When converting back to RGB the code re-maps ab to the original range.

