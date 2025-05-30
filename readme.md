
# FAS: Fast Near-Lossless ANN-SNN Conversion for Spiking LLM

This is the code implementation for the paper "FAS: Fast ANN-SNN Conversion for Spiking Large Language Models".

## Requirements


- `transformers==4.40.0.dev0`
- `accelerate >= 0.12.0`
- `torch >= 1.3`
- `datasets >= 2.14.0`
- `sentencepiece != 0.1.92`
- `protobuf`
- `evaluate`
- `scikit-learn`

## Training on NLG

### Step 1: Full-Parameter Fine-Tuning
Navigate to the directory:

```bash
cd NLG-gpt-FAS
```
Execute the following command:
```bash
export CUDA_VISIBLE_DEVICES=0,1
accelerate launch --main_process_port 41221 run_clm_FF.py \
    --model_name_or_path /src/gpt2 \
    --dataset_name HuggingFaceFW/fineweb-edu \
    --dataset_config_name sample-10BT \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --L 8 \
    --max_steps 100000 \
    --use_fast_tokenizer \
    --learning_rate 9e-5 \
    --streaming \
    --save_steps 3000 \
    --fp16 \
    --do_train \
    --output_dir /src/gpt2-pretrain
```

### Step 2: Fine-Tuning on Downstream Tasks
Execute the following command:
```bash
export CUDA_VISIBLE_DEVICES=0,1
accelerate launch --main_process_port 42222 run_clm_DT.py \
    --model_name_or_path /src/gpt2-pretrain \
    --dataset_name wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --L 8 \
    --T 8 \
    --use_fast_tokenizer \
    --learning_rate 9e-5 \
    --save_steps 2000 \
    --do_train \
    --do_eval \
    --fp16 \
    --overwrite_output_dir \
    --output_dir /src/gpt2-pretrain-fitune
```

### Step 3: Coarse-to-Fine Calibration
Execute the following command:
```bash
export CUDA_VISIBLE_DEVICES=0,1
accelerate launch --main_process_port 12429 run_clm_calib.py \
    --model_name_or_path /src/gpt2-pretrain-fitune \
    --dataset_name wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --L 8 \
    --T 8 \
    --calib_T 8 \
    --use_fast_tokenizer \
    --learning_rate 6e-5 \
    --save_steps 1500 \
    --do_train \
    --do_eval \
    --overwrite_output_dir \
    --output_dir /src/gpt2-pretrain-fitune-celib
```
## Training on NLU
### Step 1: Full-Parameter Fine-Tuning
Navigate to the directory:
```bash
cd NLU-FAS
```
Execute the following command:
```bash
export CUDA_VISIBLE_DEVICES=2,3
accelerate launch --main_process_port 41221  run_mlm.py \
    --model_name_or_path bert-base-cased \
    --dataset_name  wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --L 8 \
    --max_steps 100000 \
    --use_fast_tokenizer \
    --learning_rate 9e-5 \
    --streaming \
    --save_steps 3000 \
    --fp16 \
    --do_train \
    --logging_step 100 \
    --output_dir /src/bert-FF
```
### Step 2: Fine-Tuning on Downstream Tasks

Execute the following command:
```bash
export CUDA_VISIBLE_DEVICES=4
accelerate launch --main_process_port 41221  run_glue.py \
    --model_name_or_path /src/bert-FF \
    --task_name sst2 \
    --L 8 \
    --T 4\
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 \
    --learning_rate 3e-5 \
    --num_train_epochs 3 \
    --fp16 \
    --logging_step 10 \
    --save_steps 50000 \
    --overwrite_output_dir \
    --output_dir /src/bert-FF-DT
```
where task name can be one of cola, sst2, mrpc, stsb, qqp, mnli, qnli, rte, wnli.

### Step 3: Coarse-to-Fine Calibration

Execute the following command:
```bash
export CUDA_VISIBLE_DEVICES=0,1
accelerate launch --main_process_port 52428  run_glue_Calib.py \
  --model_name_or_path /src/bert-FF-DT \
  --task_name stsb \
  --do_train \
  --do_eval \
  --L 8 \
  --T 4 \
  --calib_T 4\
  --max_seq_length 128 \
  --per_device_train_batch_size 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --fp16 \
  --save_steps 500 \
  --overwrite_output_dir \
  --logging_step 10 \
  --output_dir /src/bert-FF-DT--calib
```

## Reference
```
@article{chen2025fas,
  title={FAS: Fast ANN-SNN Conversion for Spiking Large Language Models},
  author={Chen, Long and Song, Xiaotian and Song, Andy and Chen, BaDong and Lv, Jiancheng and Sun, Yanan},
  journal={arXiv preprint arXiv:2502.04405},
  year={2025}
}
```
