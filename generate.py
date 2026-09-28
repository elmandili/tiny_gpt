import torch

from model import TinyGPT
from tokenizer import CharacterTokenizer
from config import MODEL_PATH


device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# -------------------------
# Load checkpoint
# -------------------------

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

chars = checkpoint["chars"]


# -------------------------
# Rebuild tokenizer
# -------------------------

class LoadedTokenizer:

    def __init__(self, chars):

        self.chars = chars

        self.stoi = {
            char: i
            for i, char in enumerate(chars)
        }

        self.itos = {
            i: char
            for i, char in enumerate(chars)
        }

        self.vocab_size = len(chars)

    def encode(self, text):
        return [
            self.stoi[c]
            for c in text
        ]

    def decode(self, tokens):
        return "".join(
            self.itos[i]
            for i in tokens
        )


tokenizer = LoadedTokenizer(chars)


# -------------------------
# Load model
# -------------------------

model = TinyGPT(
    tokenizer.vocab_size
).to(device)

model.load_state_dict(
    checkpoint["model_state"]
)

model.eval()


# -------------------------
# Chat
# -------------------------

print("TinyGPT ready!")
print("Type 'quit' to exit.\n")


while True:

    user_input = input("You: ")

    if user_input.lower() == "quit":
        break

    prompt = f"User: {user_input}\nAssistant:"

    # Check unknown characters
    unknown = [
        c for c in prompt
        if c not in tokenizer.stoi
    ]

    if unknown:
        print(
            "Unknown characters:",
            set(unknown)
        )
        continue

    tokens = torch.tensor(
        [tokenizer.encode(prompt)],
        dtype=torch.long,
        device=device
    )

    generated = model.generate(
        tokens,
        max_new_tokens=300,
        tokenizer=tokenizer,
        temperature=0.8
    )

    text = tokenizer.decode(
        generated[0].tolist()
    )

    answer = text[len(prompt):]

    if "<END>" in answer:
        answer = answer.split("<END>")[0]

    answer = answer.strip()

    print("TinyGPT:", answer)
    print()