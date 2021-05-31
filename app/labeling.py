import os
import numpy as np
import pandas as pd
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from annotated_text import annotated_text
from anytree import Node, RenderTree, AsciiStyle, PreOrderIter

DATA_DIR = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "data"
)

CHECKPOINT = os.path.join(DATA_DIR, "models", "checkpoints", "checkpoint-2500")

LABELS_LIST = [
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

TAGS_MAP = {
    "B-PER": "PER",
    "I-PER": "PER",
    "B-ORG": "ORG",
    "I-ORG": "ORG",
    "B-LOC": "LOC",
    "I-LOC": "LOC",
}

TAGS = {
    "PER": "#8ef",
    "ORG": "#faa",
    "LOC": "#fea",
}

st.title("Input data:")
s = st.text_area(
    "",
    value="Germany's representative to the European Union's veterinary committee Werner Zwingmann said on Wednesday consumers should buy sheepmeat from countries other than Britain until the scientific advice was clearer.",
)


def makeTree(s: str, model_checkpoint: str):
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)

    ti = tokenizer.encode(s)
    input_ids = torch.tensor([ti])

    preds = model(input_ids)
    labels = np.argmax(preds.logits[0].detach().numpy(), axis=1)
    tokens = tokenizer.convert_ids_to_tokens(ti)

    sLower = s.lower()
    root = Node("ROOT", type="root")

    lastTag = None
    lastWord = None

    next = 0

    for i, t, l in zip(ti, tokens, labels):
        if i in tokenizer.all_special_ids:
            continue

        continuation = False

        if t.startswith("##"):
            continuation = True
            t = t[2:]

        pos = sLower.find(t[0], next)
        if pos == -1:
            raise Exception(f"Can't find '{t}' from position {next}")
        if pos > 0 and pos == next:
            continuation = True
        next = pos + len(t)
        word = s[pos:next]

        label = LABELS_LIST[l]
        tag = TAGS_MAP.get(label) or label
        if not lastTag or tag != lastTag.tag:
            lastTag = Node(f"[{tag}]", type="tag", tag=tag, parent=root)
            lastWord = Node(
                f"{'##' if continuation else ''}{word}",
                type="word",
                word=word,
                s=pos,
                e=next,
                cont=continuation,
                label=label,
                parent=lastTag,
            )
        else:
            parent = None
            if label != tag:
                if lastWord:
                    if label.startswith("I-") and lastWord.label.startswith("B-"):
                        parent = lastTag
                    elif label == lastWord.label and continuation:
                        parent = lastWord
                    else:
                        parent = lastTag = Node(
                            f"[{tag}]", type="tag", tag=tag, parent=root
                        )
                else:
                    parent = lastTag
            else:
                if continuation and lastWord:
                    parent = lastWord
                else:
                    parent = lastTag = Node(
                        f"[{tag}]", type="tag", tag=tag, parent=root
                    )

            lastWord = Node(
                f"{'##' if continuation else ''}{word}",
                type="word",
                word=word,
                s=pos,
                e=next,
                cont=continuation,
                label=label,
                parent=parent,
            )
    return root


def walkTree(tree):
    annotated = []
    interesting = []
    firstWord = True
    for t in PreOrderIter(tree, maxlevel=2, filter_=lambda n: n.type == "tag"):
        words = []
        start, end = -1, -1
        for w in PreOrderIter(t, filter_=lambda n: n.type == "word"):
            if not firstWord and not w.cont:
                words.append(" ")
            if start == -1:
                start = w.s
            end = w.e

            words.append(w.word)
            firstWord = False

        coloring = TAGS.get(t.tag)
        if coloring:
            tagged = ("".join(words), t.tag, coloring)
            annotated.append(tagged)
            interesting.append((tagged, (start, end)))
        else:
            annotated.append("".join(words))

    return annotated, interesting


tree = makeTree(s, CHECKPOINT)
annotated, interesting = walkTree(tree)

st.title("Annotated:")
annotated_text(*annotated)

st.title("Interesting:")
df = pd.DataFrame(
    [(tagged[0], tagged[1], *info) for (tagged, info) in interesting],
    columns=("Fragment", "Tag", "Start", "End"),
)
st.table(df)

st.code(RenderTree(tree, style=AsciiStyle()).by_attr())
