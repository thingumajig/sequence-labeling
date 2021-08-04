from anytree.node.node import Node
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

        tags = modelArtifact["metadata"]["tags"]
        out = []
        for i, tree in enumerate(trees):
            if isinstance(tree, Node):
                _, result = walkTree(sentences[i], tree, tags)
                out.append([(tagged[1], *info) for (tagged, info) in result])
            else:
                out.append(tree)

        return out
