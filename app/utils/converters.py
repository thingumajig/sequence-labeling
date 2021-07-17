from docx import Document


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
