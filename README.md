# Mamba-Chat 🐍

**Mamba-Chat is the first chat language model based on a state-space model architecture, not a transformer.**

The model is based on Albert Gu's and Tri Dao's work *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* ([paper](https://arxiv.org/pdf/2312.00752.pdf)) as well as their [model implementation](https://github.com/state-spaces/mamba). This repository provides training / fine-tuning code for the model based on some modifications of the Huggingface Trainer class.

Mamba-Chat is based on Mamba-2.8B and was fine-tuned on 16,000 samples of the [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) dataset. To learn more, you can:

- Take a look at the model on [Huggingface](https://huggingface.co/havenhq/mamba-chat) 🤗
- Talk to us on the [Haven](https://haven.run/) Community [Discord](https://discord.com/invite/JDjbfp6q2G) 🧑‍🤝‍🧑
- Talk to Mamba-Chat on [Google Colab](https://colab.research.google.com/drive/1dUlEYnRbgJYg4_kofNpsCddLCh6vltNK?usp=sharing)


<br>

## Run Mamba-Chat

We provide code for testing and fine-tuning our model. Here's how to get started and what you can do with it:

<br>


**Clone repository and install dependencies:**
```bash
git clone https://github.com/havenhq/mamba-chat.git
cd mamba-chat
conda create -n mamba-chat python=3.10
conda activate mamba-chat
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install accelerate transformers dataset gitpython tensorboardX datasets tensorboard scikit-learn evaluate
pip install -r requirements.txt
```

<br>

**Talk to Mamba-Chat (CLI chatbot):**
```bash
python chat.py
```

<br>

**Talk to Mamba-Chat (gradio app):**
```bash
pip install gradio==4.8.0
python app.py --share
```

<br>

**Fine-Tune Mamba (the base model) on a subset of the Ultrachat dataset:**
```bash
python train_mamba.py --model state-spaces/mamba-2.8b --tokenizer EleutherAI/gpt-neox-20b --learning_rate 5e-5 --batch_size 4 --data_path ./data/ultrachat_small.jsonl --num_epochs 3
```

<br>

**If you have a 24GB card (3090, 4090, etc.) you can use these settings:**
```bash
python train_mamba.py --model state-spaces/mamba-2.8b --tokenizer EleutherAI/gpt-neox-20b --learning_rate 5e-5 --batch_size 1 --gradient_accumulation_steps 4 --optim paged_adamw_8bit --data_path ./data/ultrachat_small.jsonl --num_epochs 3
```

**For multi-GPU**
```bash
torchrun --nproc_per_node 8 train_mamba.py --model state-spaces/mamba-130m --tokenizer EleutherAI/gpt-neox-20b --learning_rate 5e-5 --batch_size 1 --data_path ./data/ultrachat_small.jsonl --num_epochs 3
```

## Citation

```
bibtex
@misc{haven2023mambachat,
  title        = {Mamba-Chat},
  author       = {Justus Mattern and Konstantin Hohr},
  year         = {2023},
  howpublished = {GitHub},
  url          = {https://github.com/havenhq/mamba-chat}
}
```
