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

TAGS = {
    "B-PER": "#8ef",
    "I-PER": "#8ef",
    "B-ORG": "#faa",
    "I-ORG": "#faa",
    "B-LOC": "#fea",
    "I-LOC": "#fea",
}

s = st.text_area(
    "Input",
    value="Germany's representative to the European Union's veterinary committee Werner Zwingmann said on Wednesday consumers should buy sheepmeat from countries other than Britain until the scientific advice was clearer.",
)


ti = tokenizer.encode(s)
input_ids = torch.tensor([ti])

preds = model(input_ids)
labels = np.argmax(preds.logits[0].detach().numpy(), axis=1)
tokens = tokenizer.convert_ids_to_tokens(ti)

sLower = s.lower()

annotated = []
next = 0
interesting = []
for i, t, l in zip(ti, tokens, labels):
    continuation = False

    if i in tokenizer.all_special_ids:
        continue
    if t.startswith("##"):
        continuation = True
        t = t[2:]

    pos = sLower.find(t[0], next)
    if pos == -1:
        raise Exception(f"Can't find '{t}' from position {next}")
    if pos > next:
        annotated.append(" ")
    next = pos + len(t)
    word = s[pos:next]

    if continuation:
        tagged, info = last
        if isinstance(tagged, tuple):
            _w, _l, _c = tagged
            tagged = _w + word, _l, _c

            (_pos, _end) = info
            info = _pos, next
            interesting[-1] = tagged, info
        else:
            tagged = tagged + word
        annotated[-1] = tagged
        last = tagged, info
        continue

    info = (pos, next)

    tagged = None
    label = label_list[l]
    coloring = TAGS.get(label)
    if coloring:
        tagged = (word, label, coloring)
        interesting.append((tagged, info))
    tagged = tagged or word

    last = tagged, info

    annotated.append(tagged)

annotated_text(*annotated)

"""Interesting:"""
st.write([(tagged[0], tagged[1], *info) for (tagged, info) in interesting])
