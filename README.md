# EfficientGS

EfficientGS: Streamlining Gaussian Splatting for Large-Scale High-Resolution Scene Representation

## Training

```bash
sh train.sh
```

## Evaluation

```bash
python render.py -s $scene_path -m $model_path --iteration 30000 --skip_train
python metrics.py -m $model_path
```
