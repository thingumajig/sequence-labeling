import os
from typing import List, Set
import numpy as np
import pandas as pd
import streamlit as st
import torch
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForTokenClassification
from annotated_text import annotated_text
from anytree import Node, RenderTree, AsciiStyle, PreOrderIter
from docx import Document


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

# inputChoice = st.sidebar.radio(
#     "Choose input type:", ["Text input", "File input (.docx)"]
# )


def readDocx(file):
    document = Document(file)
    text = []
    for p in document.paragraphs:
        rs = p._element.xpath(".//w:t")
        text.append(" ".join([r.text for r in rs]))
    return "\n".join(text)


# if inputChoice == "Text input":
st.title("Input data:")

file = st.file_uploader("Upload a file:", ("docx",))
if file:
    value = readDocx(file)
else:
    value = "Germany's representative to the European Union's veterinary committee Werner Zwingmann said on Wednesday consumers should buy sheepmeat from countries other than Britain until the scientific advice was clearer."

s: str = st.text_area(
    "Or input text here:",
    value=value,
    height=100,
)


@st.cache(allow_output_mutation=True)
def initialize(model_checkpoint: str):
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)
    return tokenizer, model


def processText(s: str, model_checkpoint: str):
    tokenizer, model = initialize(model_checkpoint)
    sentences = sent_tokenize(s)
    st.write("Sentences:", sentences)

    special_ids = set(tokenizer.all_special_ids)
    # tis = [tokenizer.encode(sent) for sent in sentences]
    # input_ids = [torch.tensor([ti]) for ti in tis]
    # logits = [model(ii).logits[0] for ii in input_ids]

    trees = []
    globalOffset = 0
    for sent in sentences:
        ti = tokenizer.encode(sent)
        input_ids = torch.tensor([ti])
        logit = model(input_ids).logits[0]
        labels = np.argmax(logit.detach().numpy(), axis=1)
        tokens = tokenizer.convert_ids_to_tokens(ti)
        globalOffset = s.index(sent, globalOffset)
        trees.append(makeTree(sent, ti, tokens, labels, special_ids, globalOffset))
    return trees


def makeTree(
    sentence: str,
    ti: List[int],
    tokens: List[str],
    labels: List[int],
    special_ids: Set[int],
    globalOffset: int = 0,
):
    sLower = sentence.lower()
    root = Node(sentence, type="root", globalOffset=globalOffset)

    lastTag = None
    lastWord = None

    next = 0

    for i, t, l in zip(ti, tokens, labels):
        if i in special_ids:
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
        word = sentence[pos:next]

        label = LABELS_LIST[l]
        tag = TAGS_MAP.get(label) or label
        if not lastTag or tag != lastTag.tag:
            lastTag = Node(f"[{tag}]", type="tag", tag=tag, parent=root)
            lastWord = Node(
                f"{'##' if continuation else ''}{word} [{label}] [{pos + globalOffset}:{next + globalOffset}]",
                type="word",
                word=word,
                s=pos + globalOffset,
                e=next + globalOffset,
                cont=continuation,
                label=label,
                parent=lastTag,
            )
        else:
            parent = None
            if label != tag:
                if lastWord:
                    if label.startswith("I-") and lastWord.label.startswith(
                        ("B-", "I-")
                    ):
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
                f"{'##' if continuation else ''}{word} [{label}] [{pos + globalOffset}:{next + globalOffset}]",
                type="word",
                word=word,
                s=pos + globalOffset,
                e=next + globalOffset,
                cont=continuation,
                label=label,
                parent=parent,
            )
    return root


def walkTree(tree):
    annotated = []
    interesting = []
    offset = tree.globalOffset
    for t in PreOrderIter(tree, maxlevel=2, filter_=lambda n: n.type == "tag"):
        words = []
        start, end = -1, -1
        for w in PreOrderIter(t, filter_=lambda n: n.type == "word"):
            words.append(s[offset : w.s])
            if start == -1:
                start = w.s
            end = offset = w.e

            words.append(w.word)

        coloring = TAGS.get(t.tag)
        if coloring:
            subString = "".join(words)
            tagged = (subString, t.tag, coloring)
            # try:
            #     index = s.index(subString, start - 1)
            # except:
            #     st.write(subString, start, end)
            #     raise
            # if index != start:
            #     st.write(subString, "Expected:", start, "Actual:", index)
            annotated.append(tagged)
            interesting.append((tagged, (start, end)))
        else:
            annotated.append("".join(words))

    return annotated, interesting


trees = processText(s, CHECKPOINT)
annotated, interesting = [], []
for tree in trees:
    a, i = walkTree(tree)
    annotated.extend(a)
    annotated.append(" ")
    interesting.extend(i)

st.title("Annotated:")
annotated_text(*annotated, scrolling=True)

st.title("Interesting:")
df = pd.DataFrame(
    [(tagged[0], tagged[1], *info) for (tagged, info) in interesting],
    columns=("Fragment", "Tag", "Start", "End"),
)
st.table(df)

if st.checkbox("Show debug tree"):
    for tree in trees:
        st.code(RenderTree(tree, style=AsciiStyle()).by_attr())