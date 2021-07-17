import os
from pathlib import Path
from typing import Dict
from transformers import AutoTokenizer, AutoModelForTokenClassification

from app.utils.labeling import getTag
from app.bento_deploy.bento_service import TransformerService

ts = TransformerService()

DATA_DIR = Path(os.path.dirname(__file__)) / ".." / ".." / "data"
CHECKPOINT = (DATA_DIR / "models" / "checkpoints" / "checkpoint-2500").resolve()
model_checkpoint = str(CHECKPOINT)

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)

# Option 1: Pack using dictionary (recommended)
artifact = {"model": model, "tokenizer": tokenizer}
ts.pack("model", artifact)

saved_path = ts.save()
if __name__ == "__main__":
    print(saved_path)
