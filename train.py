import torch

from model import TinyGPT
from tokenizer import CharacterTokenizer

from config import (
    BATCH_SIZE,
    BLOCK_SIZE,
    LEARNING_RATE,
    MAX_STEPS,
    EVAL_INTERVAL,
    DATA_PATH,
    MODEL_PATH
)


device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# -------------------------
# Load dataset
# -------------------------

with open(
    DATA_PATH,
    "r",
    encoding="utf-8"
) as file:

    text = file.read()


print("Characters:", len(text))


# -------------------------
# Tokenizer
# -------------------------

tokenizer = CharacterTokenizer(text)

print("Vocabulary:", tokenizer.vocab_size)


data = torch.tensor(
    tokenizer.encode(text),
    dtype=torch.long
)


# 90% train / 10% validation

split = int(len(data) * 0.9)

train_data = data[:split]
validation_data = data[split:]


# -------------------------
# Batch generator
# -------------------------

def get_batch(data):

    positions = torch.randint(
        len(data) - BLOCK_SIZE - 1,
        (BATCH_SIZE,)
    )

    x = torch.stack([
        data[i:i + BLOCK_SIZE]
        for i in positions
    ])

    y = torch.stack([
        data[i + 1:i + BLOCK_SIZE + 1]
        for i in positions
    ])

    return (
        x.to(device),
        y.to(device)
    )


# -------------------------
# Model
# -------------------------

model = TinyGPT(
    tokenizer.vocab_size
).to(device)


parameters = sum(
    p.numel()
    for p in model.parameters()
)

print(
    f"Parameters: {parameters:,}"
)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# -------------------------
# Training
# -------------------------

for step in range(MAX_STEPS):

    model.train()

    x, y = get_batch(train_data)

    logits, loss = model(x, y)

    optimizer.zero_grad(
        set_to_none=True
    )

    loss.backward()

    optimizer.step()


    if step % EVAL_INTERVAL == 0:

        model.eval()

        with torch.no_grad():

            x_val, y_val = get_batch(
                validation_data
            )

            _, validation_loss = model(
                x_val,
                y_val
            )

        print(
            f"Step {step:5d} | "
            f"Train Loss: {loss.item():.4f} | "
            f"Val Loss: {validation_loss.item():.4f}"
        )


# -------------------------
# Save
# -------------------------

torch.save(
    {
        "model_state": model.state_dict(),

        "chars": tokenizer.chars
    },

    MODEL_PATH
)

print("Model saved:", MODEL_PATH)