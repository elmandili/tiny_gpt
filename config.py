# Model
BATCH_SIZE = 32
BLOCK_SIZE = 128       # Maximum context length
N_EMBD = 256           # Embedding dimension
N_HEAD = 4             # Attention heads
N_LAYER = 4            # Transformer blocks
DROPOUT = 0.1

# Training
LEARNING_RATE = 3e-4
MAX_STEPS = 5000
EVAL_INTERVAL = 250
EVAL_STEPS = 50

DATA_PATH = "data/training.txt"
MODEL_PATH = "tiny_gpt.pt"