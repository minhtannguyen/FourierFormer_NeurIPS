## Code for Language Modeling Task in Our Paper

## Requirements
This toolkit requires PyTorch `torch` and Ninja `ninja` (to compile the cuda kernels).

The experiments for the paper were conducted with Python 3.6 and PyTorch >= 1.4.0.

The toolkit supports [Weights & Biases](https://docs.wandb.ai/) for monitoring jobs. If you use it, also install `wandb`.

## Instructions

Run `sh getdata.sh` to download the data.

## Compile CUDA code

In `.src/fourier_layer-extension`, run `python setup_cuda.py install` to compile the CUDA code before running the training.

### Training

Run following commands to reproduce our results for WikiText-103 language modeling. To change the setting of R from scalar to vector, you can comment line 114 and uncomment line 113 in our `mem_transformer.py`

Softmax (small)
```
CUDA_VISIBLE_DEVICES=0,1 python ./src/train.py --cuda --data ./data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 2 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --multi_gpu --use_wandb --project_name 'fourierformer' --seed 1111 --job_name softmax-seed-1111 --work_dir ./softmax-baseline
```

Fourier (small)
```
CUDA_VISIBLE_DEVICES=0,1 python ./src/train.py --cuda --data ./data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 203 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --multi_gpu --use_wandb --project_name 'fourierformer' --seed 1111 --job_name fourier-seed_1111 --work_dir ./fourier-seed_1111 
```

Softmax (medium)
```
CUDA_VISIBLE_DEVICES=0,1 python ./src/train.py --cuda --data ./data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 256 --n_head 8 --d_head 32 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 400000 --attn_type 2 --tgt_len 384 --mem_len 0 --eval_tgt_len 384 --batch_size 56 --multi_gpu --use_wandb --project_name 'fourierformer' --seed 1111 --job_name softmax-seed-1111 --work_dir ./softmax-baseline
```

Fourier (medium)
```
CUDA_VISIBLE_DEVICES=0,1 python ./src/train.py --cuda --data ./data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 256 --n_head 8 --d_head 32 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 400000 --attn_type 203 --tgt_len 384 --mem_len 0 --eval_tgt_len 384 --batch_size 56 --multi_gpu --use_wandb --project_name 'fourierformer' --seed 1111 --job_name fourier-seed_1111 --work_dir ./fourier-seed_1111 
```