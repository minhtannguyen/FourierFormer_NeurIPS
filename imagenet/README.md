# ImagetNet with DeiT transformer

## Installation
- Python>=3.7
- Requirements:
```bash
pip install -r requirements.txt
```

## Compile CUDA code

In `./fourier_layer-extension`, run `python setup_cuda.py install` to compile the CUDA code before running the training.

## Experiments

### Baseline
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port=1 --use_env main.py --model deit_tiny_patch16_224 --batch-size 256 --data-path path/data --output_dir path/output --use_wandb --project_name 'fourier' --job_name imagenet_deit_baseline
```

### Fourier
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port=2 --use_env main.py --model deit_fourier_tiny_patch16_224 --batch-size 256 --data-path path/data --output_dir path/output --use_wandb --project_name 'fourier' --job_name imagenet_deit_fourier
```