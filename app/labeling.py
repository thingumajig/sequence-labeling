import os
import numpy as np
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from annotated_text import annotated_text


DATA_DIR = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "data"
)

model_checkpoint = os.path.join(DATA_DIR, "models", "checkpoints", "checkpoint-2500")


tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)

label_list = [
    "O",
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC",
    "B-MISC",
    "I-MISC",
]

TAGS = dict(
    (
        ("PER", "#8ef"),
        ("ORG", "#faa"),
        ("LOC", "#fea"),
    )
)

s = st.text_area(
    "Input",
    value="Germany's representative to the European Union's veterinary committee Werner Zwingmann said on Wednesday consumers should buy sheepmeat from countries other than Britain until the scientific advice was clearer.",
)


ti = tokenizer.encode(s)
input_ids = torch.tensor([ti])

preds = model(input_ids)
labels = np.argmax(preds.logits[0].detach().numpy(), axis=1)
tokens = tokenizer.convert_ids_to_tokens(ti)

annotated = []
for t, l in zip(tokens, labels):
    t = f"{t} "
    label = label_list[l]
    tagged = None
    if "-" in label:
        tag = label.split("-")[1]
        coloring = TAGS.get(tag)
        if coloring:
            tagged = (t, tag, coloring)
    tagged = tagged or t
    annotated.append(tagged)

    # st.write(f"{t}\t\t{label_list[l]}")
# st.write(annotated)
annotated_text(*annotated)
