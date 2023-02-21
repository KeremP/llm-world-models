#!/usr/bin/env python3
from posteri.embeddings import embedding
import pytest

CONTEXT = [
    "The encoder is composed of a stack of N = 6 identical layers. Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, positionwise fully connected feed-forward network. We employ a residual connection [11] around each of the two sub-layers, followed by layer normalization [1]. That is, the output of each sub-layer is LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer itself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension dmodel = 512.",
    "The decoder is also composed of a stack of N = 6 identical layers. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack. Similar to the encoder, we employ residual connections around each of the sub layers, followed by layer normalization. We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position i can depend only on the known outputs at positions less than i.",
    "During my undergraduate term at Hofstra University, I had the pleasure to work as a research and teaching assistant under Dr. Andrew Spieler – who later became my undergraduate thesis advisor – where I aided in research pertaining to Real Estate Investment Trusts, Working Capital management, and Cryptocurrencies. My work was primarily based in distilling the current literature landscape pertaining to these topics and industries into chapters to be included in related educational publications that are listed on my resume."
]

class TestEvidence(object):

    def  test_get_evidence(self, context = CONTEXT):
        
        r = embedding.extract_evidence(context)
        print(r)
        assert len(r) == 9
