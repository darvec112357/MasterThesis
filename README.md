# SMART: Self-Learning Meta-strategy Agent for Reasoning Tasks

## Description

This is the code base for the paper [SMART: Self-Learning Meta-strategy Agent for Reasoning Tasks]. It consists of two modues: inference and RL. To set up the environments, run the following commands:
```bash
pip install vllm
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps xformers trl peft accelerate bitsandbytes
```

The whole experiment can be run on one 4090. The inference engine we used is vllm and the SFT tools we used is LoRA. Feel free to adapt to other settings.

## Inference
This module is used to set up the baselines as well as create first-hand training data with few-shot learning. We used 8-shot learning throughout the experiments and manually drafted the prompts for CoT, L2M and PoT which can be found in the folder **inference/prompts**.

To run few-shot inference, run
```bash
python infer.py MODEL_NAME results/MODEL_NAME STRATEGY PART
```
where MODEL_NAME is the model you would like to use, STRATEGY is one of [cot,sub,pot] and PART is an integer from 1 to 5, it is used to run inference on MetaMathQA, which is a very big dataset and it would be better to split the dataset and run inference separately.

After obtaining the few-shot learning results, run the notebook analysis.ipynb to check the performance.

## RL
There are two sub-modules under RL, epoch1 and epoch2. In epoch1, we construct the training data using the few-shot learning results while in epoch2, we let the model generate its own training data. The detailed explanation can be found in the paper.

### Epoch 1
Run the notebook preprocess.ipynb to generate the training set, validation set and test set. After that run 
```bash
bash run.sh
```
to start fine-tuning the model. This scripts save the preprocess.ipynb and train.py for potential recovery purpose. You might need to adjust per_device_train_batch_size or gradient_accumulation_steps to avoid OOM errors. The checkpoints are saved in the directory record/DATETIME/MODELNAME, named with checkpoint-XXXXX.

Since we are using vllm for inference and the checkpoints we get are peft models, we need to convert the checkpoints to the format acceptable by vllm. This can be done by running the following commands:
```bash
git clone https://github.com/huggingface/trl.git
cd trl/examples/research_projects/stack_llama/scripts
python merge_peft_adapter.py --adapter_model_name=XXX --base_model_name=YYY
--output_name=ZZZ
```

### Epoch 2

where XXX is the path to the checkpoint model (mentioned earlier), YYY is the model you fine-tuned and ZZZ is any path you would like to store the adapted models. When you load models with vllm, you use ZZZ.

Run the first three cells of preprocess.ipynb to create the text templates. Then run
```bash
python sample.py
```
to let the model generate its own training data. Then run the rest of preprocess.ipynb to get the training data ready. The training procedure is the same as epoch 1.

Run
```bash
python evaluate.py DATASET
```
to evaluate the model's performance on the given dataset. The model path for vllm should be ZZZ as mentioned above.