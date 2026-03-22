#!/bin/bash
# Auto-generated evaluation commands for AsiaEval benchmarks
# Generated: 2026-03-21T09:02:53.216282
# Requires: pip install lm-eval
#
# Each command runs lm-evaluation-harness with --log_samples
# to produce per-item JSONL prediction files.
#
# Output structure:
#   {output_dir}/{model}/samples_{task}_{timestamp}.jsonl
#
set -e

echo '=== Command 1/44 ==='
lm_eval --model hf --model_args pretrained=openai/gpt-4o --tasks belebele_hin_Deva,belebele_ben_Beng,belebele_tam_Taml,belebele_tha_Thai,belebele_vie_Latn,belebele_ind_Latn,belebele_tgl_Latn,belebele_swa_Latn,belebele_yor_Latn,belebele_ara_Arab,belebele_zho_Hans --num_fewshot 0 --batch_size auto --output_path /lfs/skampere1/0/sttruong/torch_measure/data/asiaeval_data/raw/predictions/lm_eval_outputs/gpt-4o --log_samples

echo '=== Command 2/44 ==='
lm_eval --model hf --model_args pretrained=anthropic/claude-3.5-sonnet --tasks belebele_hin_Deva,belebele_ben_Beng,belebele_tam_Taml,belebele_tha_Thai,belebele_vie_Latn,belebele_ind_Latn,belebele_tgl_Latn,belebele_swa_Latn,belebele_yor_Latn,belebele_ara_Arab,belebele_zho_Hans --num_fewshot 0 --batch_size auto --output_path /lfs/skampere1/0/sttruong/torch_measure/data/asiaeval_data/raw/predictions/lm_eval_outputs/claude-3.5-sonnet --log_samples

echo '=== Command 3/44 ==='
lm_eval --model hf --model_args pretrained=meta-llama/Meta-Llama-3.1-70B-Instruct --tasks belebele_hin_Deva,belebele_ben_Beng,belebele_tam_Taml,belebele_tha_Thai,belebele_vie_Latn,belebele_ind_Latn,belebele_tgl_Latn,belebele_swa_Latn,belebele_yor_Latn,belebele_ara_Arab,belebele_zho_Hans --num_fewshot 0 --batch_size auto --output_path /lfs/skampere1/0/sttruong/torch_measure/data/asiaeval_data/raw/predictions/lm_eval_outputs/Meta-Llama-3.1-70B-Instruct --log_samples

echo '=== Command 4/44 ==='
lm_eval --model hf --model_args pretrained=Qwen/Qwen2.5-72B-Instruct --tasks belebele_hin_Deva,belebele_ben_Beng,belebele_tam_Taml,belebele_tha_Thai,belebele_vie_Latn,belebele_ind_Latn,belebele_tgl_Latn,belebele_swa_Latn,belebele_yor_Latn,belebele_ara_Arab,belebele_zho_Hans --num_fewshot 0 --batch_size auto --output_path /lfs/skampere1/0/sttruong/torch_measure/data/asiaeval_data/raw/predictions/lm_eval_outputs/Qwen2.5-72B-Instruct --log_samples

echo '=== Command 5/44 ==='
lm_eval --model hf --model_args pretrained=google/gemma-2-27b-it --tasks belebele_hin_Deva,belebele_ben_Beng,belebele_tam_Taml,belebele_tha_Thai,belebele_vie_Latn,belebele_ind_Latn,belebele_tgl_Latn,belebele_swa_Latn,belebele_yor_Latn,belebele_ara_Arab,belebele_zho_Hans --num_fewshot 0 --batch_size auto --output_path /lfs/skampere1/0/sttruong/torch_measure/data/asiaeval_data/raw/predictions/lm_eval_outputs/gemma-2-27b-it --log_samples

echo '=== Command 6/44 ==='
lm_eval --model hf --model_args pretrained=meta-llama/Meta-Llama-3.1-8B-Instruct --tasks belebele_hin_Deva,belebele_ben_Beng,belebele_tam_Taml,belebele_tha_Thai,belebele_vie_Latn,belebele_ind_Latn,belebele_tgl_Latn,belebele_swa_Latn,belebele_yor_Latn,belebele_ara_Arab,belebele_zho_Hans --num_fewshot 0 --batch_size auto --output_path /lfs/skampere1/0/sttruong/torch_measure/data/asiaeval_data/raw/predictions/lm_eval_outputs/Meta-Llama-3.1-8B-Instruct --log_samples

echo '=== Command 7/44 ==='
lm_eval --model hf --model_args pretrained=Qwen/Qwen2.5-7B-Instruct --tasks belebele_hin_Deva,belebele_ben_Beng,belebele_tam_Taml,belebele_tha_Thai,belebele_vie_Latn,belebele_ind_Latn,belebele_tgl_Latn,belebele_swa_Latn,belebele_yor_Latn,belebele_ara_Arab,belebele_zho_Hans --num_fewshot 0 --batch_size auto --output_path /lfs/skampere1/0/sttruong/torch_measure/data/asiaeval_data/raw/predictions/lm_eval_outputs/Qwen2.5-7B-Instruct --log_samples

echo '=== Command 8/44 ==='
lm_eval --model hf --model_args pretrained=google/gemma-2-9b-it --tasks belebele_hin_Deva,belebele_ben_Beng,belebele_tam_Taml,belebele_tha_Thai,belebele_vie_Latn,belebele_ind_Latn,belebele_tgl_Latn,belebele_swa_Latn,belebele_yor_Latn,belebele_ara_Arab,belebele_zho_Hans --num_fewshot 0 --batch_size auto --output_path /lfs/skampere1/0/sttruong/torch_measure/data/asiaeval_data/raw/predictions/lm_eval_outputs/gemma-2-9b-it --log_samples

echo '=== Command 9/44 ==='
lm_eval --model hf --model_args pretrained=aisingapore/Gemma-SEA-LION-v3-9B --tasks belebele_hin_Deva,belebele_ben_Beng,belebele_tam_Taml,belebele_tha_Thai,belebele_vie_Latn,belebele_ind_Latn,belebele_tgl_Latn,belebele_swa_Latn,belebele_yor_Latn,belebele_ara_Arab,belebele_zho_Hans --num_fewshot 0 --batch_size auto --output_path /lfs/skampere1/0/sttruong/torch_measure/data/asiaeval_data/raw/predictions/lm_eval_outputs/Gemma-SEA-LION-v3-9B --log_samples

echo '=== Command 10/44 ==='
lm_eval --model hf --model_args pretrained=scb10x/typhoon-v1.5-7b-instruct --tasks belebele_hin_Deva,belebele_ben_Beng,belebele_tam_Taml,belebele_tha_Thai,belebele_vie_Latn,belebele_ind_Latn,belebele_tgl_Latn,belebele_swa_Latn,belebele_yor_Latn,belebele_ara_Arab,belebele_zho_Hans --num_fewshot 0 --batch_size auto --output_path /lfs/skampere1/0/sttruong/torch_measure/data/asiaeval_data/raw/predictions/lm_eval_outputs/typhoon-v1.5-7b-instruct --log_samples

echo '=== Command 11/44 ==='
lm_eval --model hf --model_args pretrained=ai4bharat/Airavata --tasks belebele_hin_Deva,belebele_ben_Beng,belebele_tam_Taml,belebele_tha_Thai,belebele_vie_Latn,belebele_ind_Latn,belebele_tgl_Latn,belebele_swa_Latn,belebele_yor_Latn,belebele_ara_Arab,belebele_zho_Hans --num_fewshot 0 --batch_size auto --output_path /lfs/skampere1/0/sttruong/torch_measure/data/asiaeval_data/raw/predictions/lm_eval_outputs/Airavata --log_samples

echo '=== Command 12/44 ==='
lm_eval --model hf --model_args pretrained=openai/gpt-4o --tasks xcopa_et,xcopa_ht,xcopa_id,xcopa_it,xcopa_qu,xcopa_sw,xcopa_ta,xcopa_th,xcopa_tr,xcopa_vi,xcopa_zh --num_fewshot 0 --batch_size auto --output_path /lfs/skampere1/0/sttruong/torch_measure/data/asiaeval_data/raw/predictions/lm_eval_outputs/gpt-4o --log_samples

echo '=== Command 13/44 ==='
lm_eval --model hf --model_args pretrained=anthropic/claude-3.5-sonnet --tasks xcopa_et,xcopa_ht,xcopa_id,xcopa_it,xcopa_qu,xcopa_sw,xcopa_ta,xcopa_th,xcopa_tr,xcopa_vi,xcopa_zh --num_fewshot 0 --batch_size auto --output_path /lfs/skampere1/0/sttruong/torch_measure/data/asiaeval_data/raw/predictions/lm_eval_outputs/claude-3.5-sonnet --log_samples

echo '=== Command 14/44 ==='
lm_eval --model hf --model_args pretrained=meta-llama/Meta-Llama-3.1-70B-Instruct --tasks xcopa_et,xcopa_ht,xcopa_id,xcopa_it,xcopa_qu,xcopa_sw,xcopa_ta,xcopa_th,xcopa_tr,xcopa_vi,xcopa_zh --num_fewshot 0 --batch_size auto --output_path /lfs/skampere1/0/sttruong/torch_measure/data/asiaeval_data/raw/predictions/lm_eval_outputs/Meta-Llama-3.1-70B-Instruct --log_samples

echo '=== Command 15/44 ==='
lm_eval --model hf --model_args pretrained=Qwen/Qwen2.5-72B-Instruct --tasks xcopa_et,xcopa_ht,xcopa_id,xcopa_it,xcopa_qu,xcopa_sw,xcopa_ta,xcopa_th,xcopa_tr,xcopa_vi,xcopa_zh --num_fewshot 0 --batch_size auto --output_path /lfs/skampere1/0/sttruong/torch_measure/data/asiaeval_data/raw/predictions/lm_eval_outputs/Qwen2.5-72B-Instruct --log_samples

echo '=== Command 16/44 ==='
lm_eval --model hf --model_args pretrained=google/gemma-2-27b-it --tasks xcopa_et,xcopa_ht,xcopa_id,xcopa_it,xcopa_qu,xcopa_sw,xcopa_ta,xcopa_th,xcopa_tr,xcopa_vi,xcopa_zh --num_fewshot 0 --batch_size auto --output_path /lfs/skampere1/0/sttruong/torch_measure/data/asiaeval_data/raw/predictions/lm_eval_outputs/gemma-2-27b-it --log_samples

echo '=== Command 17/44 ==='
lm_eval --model hf --model_args pretrained=meta-llama/Meta-Llama-3.1-8B-Instruct --tasks xcopa_et,xcopa_ht,xcopa_id,xcopa_it,xcopa_qu,xcopa_sw,xcopa_ta,xcopa_th,xcopa_tr,xcopa_vi,xcopa_zh --num_fewshot 0 --batch_size auto --output_path /lfs/skampere1/0/sttruong/torch_measure/data/asiaeval_data/raw/predictions/lm_eval_outputs/Meta-Llama-3.1-8B-Instruct --log_samples

echo '=== Command 18/44 ==='
lm_eval --model hf --model_args pretrained=Qwen/Qwen2.5-7B-Instruct --tasks xcopa_et,xcopa_ht,xcopa_id,xcopa_it,xcopa_qu,xcopa_sw,xcopa_ta,xcopa_th,xcopa_tr,xcopa_vi,xcopa_zh --num_fewshot 0 --batch_size auto --output_path /lfs/skampere1/0/sttruong/torch_measure/data/asiaeval_data/raw/predictions/lm_eval_outputs/Qwen2.5-7B-Instruct --log_samples

echo '=== Command 19/44 ==='
lm_eval --model hf --model_args pretrained=google/gemma-2-9b-it --tasks xcopa_et,xcopa_ht,xcopa_id,xcopa_it,xcopa_qu,xcopa_sw,xcopa_ta,xcopa_th,xcopa_tr,xcopa_vi,xcopa_zh --num_fewshot 0 --batch_size auto --output_path /lfs/skampere1/0/sttruong/torch_measure/data/asiaeval_data/raw/predictions/lm_eval_outputs/gemma-2-9b-it --log_samples

echo '=== Command 20/44 ==='
lm_eval --model hf --model_args pretrained=aisingapore/Gemma-SEA-LION-v3-9B --tasks xcopa_et,xcopa_ht,xcopa_id,xcopa_it,xcopa_qu,xcopa_sw,xcopa_ta,xcopa_th,xcopa_tr,xcopa_vi,xcopa_zh --num_fewshot 0 --batch_size auto --output_path /lfs/skampere1/0/sttruong/torch_measure/data/asiaeval_data/raw/predictions/lm_eval_outputs/Gemma-SEA-LION-v3-9B --log_samples

echo '=== Command 21/44 ==='
lm_eval --model hf --model_args pretrained=scb10x/typhoon-v1.5-7b-instruct --tasks xcopa_et,xcopa_ht,xcopa_id,xcopa_it,xcopa_qu,xcopa_sw,xcopa_ta,xcopa_th,xcopa_tr,xcopa_vi,xcopa_zh --num_fewshot 0 --batch_size auto --output_path /lfs/skampere1/0/sttruong/torch_measure/data/asiaeval_data/raw/predictions/lm_eval_outputs/typhoon-v1.5-7b-instruct --log_samples

echo '=== Command 22/44 ==='
lm_eval --model hf --model_args pretrained=ai4bharat/Airavata --tasks xcopa_et,xcopa_ht,xcopa_id,xcopa_it,xcopa_qu,xcopa_sw,xcopa_ta,xcopa_th,xcopa_tr,xcopa_vi,xcopa_zh --num_fewshot 0 --batch_size auto --output_path /lfs/skampere1/0/sttruong/torch_measure/data/asiaeval_data/raw/predictions/lm_eval_outputs/Airavata --log_samples

echo '=== Command 23/44 ==='
lm_eval --model hf --model_args pretrained=openai/gpt-4o --tasks global_mmlu_ar,global_mmlu_hi,global_mmlu_bn,global_mmlu_th,global_mmlu_vi,global_mmlu_id,global_mmlu_sw,global_mmlu_yo,global_mmlu_es,global_mmlu_pt,global_mmlu_zh --num_fewshot 0 --batch_size auto --output_path /lfs/skampere1/0/sttruong/torch_measure/data/asiaeval_data/raw/predictions/lm_eval_outputs/gpt-4o --log_samples

echo '=== Command 24/44 ==='
lm_eval --model hf --model_args pretrained=anthropic/claude-3.5-sonnet --tasks global_mmlu_ar,global_mmlu_hi,global_mmlu_bn,global_mmlu_th,global_mmlu_vi,global_mmlu_id,global_mmlu_sw,global_mmlu_yo,global_mmlu_es,global_mmlu_pt,global_mmlu_zh --num_fewshot 0 --batch_size auto --output_path /lfs/skampere1/0/sttruong/torch_measure/data/asiaeval_data/raw/predictions/lm_eval_outputs/claude-3.5-sonnet --log_samples

echo '=== Command 25/44 ==='
lm_eval --model hf --model_args pretrained=meta-llama/Meta-Llama-3.1-70B-Instruct --tasks global_mmlu_ar,global_mmlu_hi,global_mmlu_bn,global_mmlu_th,global_mmlu_vi,global_mmlu_id,global_mmlu_sw,global_mmlu_yo,global_mmlu_es,global_mmlu_pt,global_mmlu_zh --num_fewshot 0 --batch_size auto --output_path /lfs/skampere1/0/sttruong/torch_measure/data/asiaeval_data/raw/predictions/lm_eval_outputs/Meta-Llama-3.1-70B-Instruct --log_samples

echo '=== Command 26/44 ==='
lm_eval --model hf --model_args pretrained=Qwen/Qwen2.5-72B-Instruct --tasks global_mmlu_ar,global_mmlu_hi,global_mmlu_bn,global_mmlu_th,global_mmlu_vi,global_mmlu_id,global_mmlu_sw,global_mmlu_yo,global_mmlu_es,global_mmlu_pt,global_mmlu_zh --num_fewshot 0 --batch_size auto --output_path /lfs/skampere1/0/sttruong/torch_measure/data/asiaeval_data/raw/predictions/lm_eval_outputs/Qwen2.5-72B-Instruct --log_samples

echo '=== Command 27/44 ==='
lm_eval --model hf --model_args pretrained=google/gemma-2-27b-it --tasks global_mmlu_ar,global_mmlu_hi,global_mmlu_bn,global_mmlu_th,global_mmlu_vi,global_mmlu_id,global_mmlu_sw,global_mmlu_yo,global_mmlu_es,global_mmlu_pt,global_mmlu_zh --num_fewshot 0 --batch_size auto --output_path /lfs/skampere1/0/sttruong/torch_measure/data/asiaeval_data/raw/predictions/lm_eval_outputs/gemma-2-27b-it --log_samples

echo '=== Command 28/44 ==='
lm_eval --model hf --model_args pretrained=meta-llama/Meta-Llama-3.1-8B-Instruct --tasks global_mmlu_ar,global_mmlu_hi,global_mmlu_bn,global_mmlu_th,global_mmlu_vi,global_mmlu_id,global_mmlu_sw,global_mmlu_yo,global_mmlu_es,global_mmlu_pt,global_mmlu_zh --num_fewshot 0 --batch_size auto --output_path /lfs/skampere1/0/sttruong/torch_measure/data/asiaeval_data/raw/predictions/lm_eval_outputs/Meta-Llama-3.1-8B-Instruct --log_samples

echo '=== Command 29/44 ==='
lm_eval --model hf --model_args pretrained=Qwen/Qwen2.5-7B-Instruct --tasks global_mmlu_ar,global_mmlu_hi,global_mmlu_bn,global_mmlu_th,global_mmlu_vi,global_mmlu_id,global_mmlu_sw,global_mmlu_yo,global_mmlu_es,global_mmlu_pt,global_mmlu_zh --num_fewshot 0 --batch_size auto --output_path /lfs/skampere1/0/sttruong/torch_measure/data/asiaeval_data/raw/predictions/lm_eval_outputs/Qwen2.5-7B-Instruct --log_samples

echo '=== Command 30/44 ==='
lm_eval --model hf --model_args pretrained=google/gemma-2-9b-it --tasks global_mmlu_ar,global_mmlu_hi,global_mmlu_bn,global_mmlu_th,global_mmlu_vi,global_mmlu_id,global_mmlu_sw,global_mmlu_yo,global_mmlu_es,global_mmlu_pt,global_mmlu_zh --num_fewshot 0 --batch_size auto --output_path /lfs/skampere1/0/sttruong/torch_measure/data/asiaeval_data/raw/predictions/lm_eval_outputs/gemma-2-9b-it --log_samples

echo '=== Command 31/44 ==='
lm_eval --model hf --model_args pretrained=aisingapore/Gemma-SEA-LION-v3-9B --tasks global_mmlu_ar,global_mmlu_hi,global_mmlu_bn,global_mmlu_th,global_mmlu_vi,global_mmlu_id,global_mmlu_sw,global_mmlu_yo,global_mmlu_es,global_mmlu_pt,global_mmlu_zh --num_fewshot 0 --batch_size auto --output_path /lfs/skampere1/0/sttruong/torch_measure/data/asiaeval_data/raw/predictions/lm_eval_outputs/Gemma-SEA-LION-v3-9B --log_samples

echo '=== Command 32/44 ==='
lm_eval --model hf --model_args pretrained=scb10x/typhoon-v1.5-7b-instruct --tasks global_mmlu_ar,global_mmlu_hi,global_mmlu_bn,global_mmlu_th,global_mmlu_vi,global_mmlu_id,global_mmlu_sw,global_mmlu_yo,global_mmlu_es,global_mmlu_pt,global_mmlu_zh --num_fewshot 0 --batch_size auto --output_path /lfs/skampere1/0/sttruong/torch_measure/data/asiaeval_data/raw/predictions/lm_eval_outputs/typhoon-v1.5-7b-instruct --log_samples

echo '=== Command 33/44 ==='
lm_eval --model hf --model_args pretrained=ai4bharat/Airavata --tasks global_mmlu_ar,global_mmlu_hi,global_mmlu_bn,global_mmlu_th,global_mmlu_vi,global_mmlu_id,global_mmlu_sw,global_mmlu_yo,global_mmlu_es,global_mmlu_pt,global_mmlu_zh --num_fewshot 0 --batch_size auto --output_path /lfs/skampere1/0/sttruong/torch_measure/data/asiaeval_data/raw/predictions/lm_eval_outputs/Airavata --log_samples

echo '=== Command 34/44 ==='
lm_eval --model hf --model_args pretrained=openai/gpt-4o --tasks indommlu --num_fewshot 0 --batch_size auto --output_path /lfs/skampere1/0/sttruong/torch_measure/data/asiaeval_data/raw/predictions/lm_eval_outputs/gpt-4o --log_samples

echo '=== Command 35/44 ==='
lm_eval --model hf --model_args pretrained=anthropic/claude-3.5-sonnet --tasks indommlu --num_fewshot 0 --batch_size auto --output_path /lfs/skampere1/0/sttruong/torch_measure/data/asiaeval_data/raw/predictions/lm_eval_outputs/claude-3.5-sonnet --log_samples

echo '=== Command 36/44 ==='
lm_eval --model hf --model_args pretrained=meta-llama/Meta-Llama-3.1-70B-Instruct --tasks indommlu --num_fewshot 0 --batch_size auto --output_path /lfs/skampere1/0/sttruong/torch_measure/data/asiaeval_data/raw/predictions/lm_eval_outputs/Meta-Llama-3.1-70B-Instruct --log_samples

echo '=== Command 37/44 ==='
lm_eval --model hf --model_args pretrained=Qwen/Qwen2.5-72B-Instruct --tasks indommlu --num_fewshot 0 --batch_size auto --output_path /lfs/skampere1/0/sttruong/torch_measure/data/asiaeval_data/raw/predictions/lm_eval_outputs/Qwen2.5-72B-Instruct --log_samples

echo '=== Command 38/44 ==='
lm_eval --model hf --model_args pretrained=google/gemma-2-27b-it --tasks indommlu --num_fewshot 0 --batch_size auto --output_path /lfs/skampere1/0/sttruong/torch_measure/data/asiaeval_data/raw/predictions/lm_eval_outputs/gemma-2-27b-it --log_samples

echo '=== Command 39/44 ==='
lm_eval --model hf --model_args pretrained=meta-llama/Meta-Llama-3.1-8B-Instruct --tasks indommlu --num_fewshot 0 --batch_size auto --output_path /lfs/skampere1/0/sttruong/torch_measure/data/asiaeval_data/raw/predictions/lm_eval_outputs/Meta-Llama-3.1-8B-Instruct --log_samples

echo '=== Command 40/44 ==='
lm_eval --model hf --model_args pretrained=Qwen/Qwen2.5-7B-Instruct --tasks indommlu --num_fewshot 0 --batch_size auto --output_path /lfs/skampere1/0/sttruong/torch_measure/data/asiaeval_data/raw/predictions/lm_eval_outputs/Qwen2.5-7B-Instruct --log_samples

echo '=== Command 41/44 ==='
lm_eval --model hf --model_args pretrained=google/gemma-2-9b-it --tasks indommlu --num_fewshot 0 --batch_size auto --output_path /lfs/skampere1/0/sttruong/torch_measure/data/asiaeval_data/raw/predictions/lm_eval_outputs/gemma-2-9b-it --log_samples

echo '=== Command 42/44 ==='
lm_eval --model hf --model_args pretrained=aisingapore/Gemma-SEA-LION-v3-9B --tasks indommlu --num_fewshot 0 --batch_size auto --output_path /lfs/skampere1/0/sttruong/torch_measure/data/asiaeval_data/raw/predictions/lm_eval_outputs/Gemma-SEA-LION-v3-9B --log_samples

echo '=== Command 43/44 ==='
lm_eval --model hf --model_args pretrained=scb10x/typhoon-v1.5-7b-instruct --tasks indommlu --num_fewshot 0 --batch_size auto --output_path /lfs/skampere1/0/sttruong/torch_measure/data/asiaeval_data/raw/predictions/lm_eval_outputs/typhoon-v1.5-7b-instruct --log_samples

echo '=== Command 44/44 ==='
lm_eval --model hf --model_args pretrained=ai4bharat/Airavata --tasks indommlu --num_fewshot 0 --batch_size auto --output_path /lfs/skampere1/0/sttruong/torch_measure/data/asiaeval_data/raw/predictions/lm_eval_outputs/Airavata --log_samples

