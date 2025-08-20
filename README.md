# Transformer Cartpole

Transformer applied to Cartpole. Implementation is a bit jank as batch_size is used as seq_len (i.e. batching isn't implemented) but it learns to win indefinitely. Baseline DQN adapted from https://raw.githubusercontent.com/seungeunrho/minimalRL/refs/heads/master/dqn.py. Transformer adapted from https://raw.githubusercontent.com/karpathy/nanoGPT/refs/heads/master/model.py.

## Usage
```
python3.10 attention_cartpole.py
```
