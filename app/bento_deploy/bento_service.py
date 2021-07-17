from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import JsonInput
from bentoml.frameworks.transformers import TransformersModelArtifact

from app.utils.labeling import prepareArtifact, processSentence, walkTree


@env(pip_packages=["transformers", "torch", "numpy", "anytree", "nltk"])
@artifacts([TransformersModelArtifact("model")])
class TransformerService(BentoService):
    @api(input=JsonInput(), batch=False)
    def detect(self, sentences):
        modelArtifact = prepareArtifact(
            self.artifacts.model["tokenizer"], self.artifacts.model["model"]
        )
        trees = []
        for sentence in sentences:
            trees.append(processSentence(sentence, modelArtifact, sentence))

        interesting = []
        tags = modelArtifact["metadata"]["tags"]
        for i, tree in enumerate(trees):
            _, result = walkTree(sentences[i], tree, tags)
            interesting.extend(result)

        return [(tagged[1], *info) for (tagged, info) in interesting]
