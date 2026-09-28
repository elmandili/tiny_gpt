class CharacterTokenizer:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))

        self.vocab_size = len(self.chars)

        self.stoi = {
            char: index
            for index, char in enumerate(self.chars)
        }

        self.itos = {
            index: char
            for index, char in enumerate(self.chars)
        }

    def encode(self, text):
        return [self.stoi[c] for c in text]

    def decode(self, tokens):
        return "".join(self.itos[i] for i in tokens)