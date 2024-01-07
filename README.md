# LoRA replication using Jittor
This repository contains the implementation of [LoRA](https://arxiv.org/abs/2106.09685) using the Jittor framework. LoRA is used to adapt a pre-trained model to new tasks or domains by leveraging low-rank structures within the model's parameters.

## Model Performance
| Model & Method                 | BLEU            | NIST           | MET            | ROUGE-L         | CIDEr |
|-------------------------|-----------------|----------------|----------------|-----------------|-------|
| Original GPT-2 M (LoRA) | $70.4_{\pm.1}$ | $8.85_{\pm.02}$ | $46.8_{\pm.2}$ | $71.8_{\pm.1}$ | $2.53_{\pm.02}$ |
| Original GPT-2 L (LoRA) | $70.4_{\pm.1}$ | $8.89_{\pm.02}$ | $46.8_{\pm.2}$ | $72.0_{\pm.2}$ | $2.47_{\pm.02}$ |
| Jittor GPT2-M  | 68.72 | 8.66   | 46.20 | 71.30 | 2.50   |
| Jittor GPT2-L  | 69.75 | 8.82   | 46.44 | 71.27 | 2.52   |

Similar to the LoRA paper, we added LoRA layers to the W matrices in the attention module, i.e. W' = W + AB. Due to limitation of computational resources, we used a smaller batch size to train the model, which is an explanation why there is a small difference in the metric scores compared to the results in the paper.
For BoolQ, the GPT2-medium model has an accuracy of 71.8%, and the GPT2-large model has an accuracy of 73.9%, which both surpasses the accuracy of GPT-3 (zero-shot) at 60.2%.

## Requirements
- Python (>=3.7)
- Jittor (1.3.8.5)
- NumPy
- wandb (for logging)
- (Optional) CUDA-compatible GPU for faster computation


## Pretrained Models
Please refer to https://github.com/microsoft/LoRA/blob/main/examples/NLG/download_pretrained_checkpoints.sh to obtain the pretrained checkpoints.

## Usage
Below is an example on how to use the lora modules.
```
self.c_attn = mylora_layers.MergedLinear(
            nx, n_state * 3, 
            r=config.lora_attn_dim, 
            lora_alpha=config.lora_attn_alpha, 
            lora_dropout=config.lora_dropout, 
            enable_lora=[True, False, True], 
            fan_in_fan_out=True,
            merge_weights=False
        )
```
To run the e2e replication, 
1. git clone `https://github.com/microsoft/LoRA.git` to obtain the dataset and some utils and scripts
2. use `LoRA/examples/NLG/create_datasets.sh` to preprocess the dataset
3. download the pretrained models using `LoRA/examples/NLG/download_pretrained_checkpoints.sh`
4. set the paths for `--train_data`, `--valid_data`, `init_checkpoint` in `finetune.sh` and run the script
    - note: please adjust `--train_batch_size` accordingly to avoid gpu memory overflow
    - also, to obtain logging, configure wandb in advance
5. set the path for `init_checkpoint` in `run_inference.sh` to determine the checkpoint to be used to generate the outputs on e2e
6. set the path of the output and refernce using `--sample_file` and `--input_file` in `run_metric.sh` and run the script to obtain the results

For BoolQ,
1. (download)[https://github.com/google-research-datasets/boolean-questions] and preprocess the data using `change_key.py`
2. split the data into test, valid, train
3. use the same procedures as e2e to finetune and generate the output (using `finetune.sh` and `run_inference.sh`)
4. use `eval_boolq.py` to get the metrics

## Reference
- https://arxiv.org/abs/2106.09685
- https://github.com/microsoft/LoRA/tree/main