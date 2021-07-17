from typing import Any, Dict, List, Optional, Set, TypedDict

from anytree import Node, PreOrderIter
from nltk.tokenize import sent_tokenize
import torch
import numpy as np


def getTag(label):
    return label.replace("B-", "").replace("I-", "")


class ModelArtifactMetadata(TypedDict):
    ignoreLabels: Set[str]
    labelsDict: Dict[int, str]
    tagsMap: Dict[str, str]
    tags: Dict[str, str]
    specialIds: Set[int]


class ModelArtifact(TypedDict):
    model: Any
    tokenizer: Any
    metadata: ModelArtifactMetadata


def prepareArtifact(tokenizer, model, tagMapper=lambda tag: tag):
    config = model.config

    ignoreLabels = {"O", "B-MISC", "I-MISC"}
    labelsDict = {k: v for (k, v) in config.id2label.items()}
    tagsMap = {l: getTag(l) for l in labelsDict.values() if l not in ignoreLabels}
    tags = {t: tagMapper(t) for t in tagsMap.values()}
    specialIds = set(tokenizer.all_special_ids)

    return ModelArtifact(
        tokenizer=tokenizer,
        model=model,
        metadata=ModelArtifactMetadata(
            ignoreLabels=ignoreLabels,
            labelsDict=labelsDict,
            tagsMap=tagsMap,
            tags=tags,
            specialIds=specialIds,
        ),
    )


def processText(s: str, modelArtifact: ModelArtifact):
    sentences = sent_tokenize(s)

    # tis = [
    #     tokenizer.encode(sent, padding="max_length") for sent in sentences
    # ]
    # input_ids = torch.tensor(tis)
    # logits = model(input_ids).logits

    trees = []
    globalOffset = 0
    for i, sent in enumerate(sentences):
        trees.append(processSentence(sent, modelArtifact, s, globalOffset))
    return trees, sentences


def processSentence(sentence, modelArtifact: ModelArtifact, text=None, globalOffset=0):
    tokenizer = modelArtifact["tokenizer"]
    model = modelArtifact["model"]
    metadata = modelArtifact["metadata"]
    ti = tokenizer.encode(sentence)
    input_ids = torch.tensor([ti])
    logit = model(input_ids).logits[0]
    # ti = tis[i]
    # logit = logits[i]
    labels = np.argmax(logit.detach().numpy(), axis=1)
    tokens = tokenizer.convert_ids_to_tokens(ti)
    if text:
        globalOffset = text.index(sentence, globalOffset)
    metadata = metadata
    return makeTree(
        sentence,
        ti,
        tokens,
        labels,
        metadata["specialIds"],
        metadata["labelsDict"],
        metadata["tagsMap"],
        globalOffset,
    )


def makeTree(
    sentence: str,
    ti: List[int],
    tokens: List[str],
    labels: List[int],
    specialIds: Set[int],
    labelsDict: Dict[int, str],
    tagsMap: Dict[str, str],
    globalOffset: int = 0,
):
    sLower = sentence.lower()
    root = Node(sentence, type="root", globalOffset=globalOffset)

    lastTag: Optional[Node] = None
    lastWord: Optional[Node] = None

    next = 0

    for i, t, l in zip(ti, tokens, labels):
        if i in specialIds:
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

        label = labelsDict[l]
        tag = tagsMap.get(label) or label
        if not lastTag or tag != lastTag.tag:  # type: ignore
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
                    if label.startswith("I-") and lastWord.label.startswith(  # type: ignore
                        ("B-", "I-")
                    ):
                        parent = lastTag
                    elif label == lastWord.label and continuation:  # type: ignore
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


def walkTree(text, tree: Node, tags: Dict[str, str]):
    annotated = []
    interesting = []
    offset = tree.globalOffset  # type: ignore
    for t in PreOrderIter(tree, maxlevel=2, filter_=lambda n: n.type == "tag"):
        words = []
        start, end = -1, -1
        for w in PreOrderIter(t, filter_=lambda n: n.type == "word"):
            words.append(text[offset : w.s])
            if start == -1:
                start = w.s
            end = offset = w.e

            words.append(w.word)

        coloring = tags.get(t.tag)
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
