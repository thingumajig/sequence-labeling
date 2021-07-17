from app.utils.labeling import (
    ModelArtifact,
    ModelArtifactMetadata,
    prepareArtifact,
    processText,
    walkTree,
)
import pandas as pd
import streamlit as st
from transformers import AutoTokenizer, AutoModelForTokenClassification
from annotated_text import annotated_text
from anytree import RenderTree, AsciiStyle
from docx import Document
from pathlib import Path
from app.utils.color import getColorForText


st.set_page_config(
    "NER DEMO", page_icon="🏷", layout="wide", initial_sidebar_state="auto"
)

DATA_DIR = Path.cwd() / "data"
CHECKPOINT = DATA_DIR / "models" / "checkpoints" / "checkpoint-2500"

DEFAULT_TEXT = "Germany's representative to the European Union's veterinary committee Werner Zwingmann said on Wednesday consumers should buy sheepmeat from countries other than Britain until the scientific advice was clearer."


def getTag(label):
    return label.replace("B-", "").replace("I-", "")


@st.cache(allow_output_mutation=True, show_spinner=True)
def initialize(model_checkpoint: str):
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)
    return prepareArtifact(tokenizer, model, tagMapper=getColorForText)


modelArtifact = initialize(str(CHECKPOINT))


def readDocx(file):
    document = Document(file)
    text = []
    for p in document.paragraphs:
        pText = []
        rs = p._element.xpath(".//w:t")
        if rs:
            pText.append("".join([r.text for r in rs]))
        if pText:
            pText[-1] += "."
            text.append("".join(pText))
    return "\n".join(text)


# if inputChoice == "Text input":
st.title("Input data:")

file = st.file_uploader("Upload a file:", ("docx",))
if file:
    with st.spinner(text="Extracting text..."):
        value = readDocx(file)
else:
    value = DEFAULT_TEXT

text: str = st.text_area(
    "Or input text here:" if not file else "",
    value=value,
    height=100,
)

with st.spinner(text="Labeling sequence:"):
    trees, sentences = processText(text, modelArtifact)

with st.spinner(text="Rendering results..."):
    if st.sidebar.checkbox("Show sentences"):
        st.write("Sentences:", sentences)

    annotated, interesting = [], []
    tags = modelArtifact["metadata"]["tags"]
    for tree in trees:
        a, i = walkTree(text, tree, tags)
        annotated.extend(a)
        annotated.append(" ")
        interesting.extend(i)

    st.title("Annotated:")
    annotated_text(*annotated, scrolling=True, height=300)

    if st.sidebar.checkbox("Show Interesting Block"):
        st.title("Interesting:")
        df = pd.DataFrame(
            [(tagged[0], tagged[1], *info) for (tagged, info) in interesting],
            columns=("Fragment", "Tag", "Start", "End"),
        )
        st.table(df)

    if st.sidebar.checkbox("Show debug tree"):
        for tree in trees:
            st.code(RenderTree(tree, style=AsciiStyle()).by_attr())
