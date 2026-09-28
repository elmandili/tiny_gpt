from datasets import load_dataset

dataset = load_dataset("roskoN/dailydialog")

with open("data/training.txt", "w", encoding="utf-8") as f:

    for conversation in dataset["train"]:

        utterances = conversation["utterances"]

        for i, message in enumerate(utterances):

            

            f.write(f"{message}<END>\n")

        f.write("\n")

print("Done!")