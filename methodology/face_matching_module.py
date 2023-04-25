from torch import nn


class FaceMatching(nn.Module):
    def __init__(self, pretrained_model, name):
        super(FaceMatching, self).__init__()
        self.pretrained = pretrained_model
        self.cosine_layer = nn.Linear(512, 1, False)
        self.name = name

    def forward(self, x):
        if self.name == 'CurricularFace':
            x = self.pretrained(x)[0]
        else:  # self.name in ['ElasticArcface', 'ElasticCosface']:
            x = self.pretrained(x)
        x = nn.functional.normalize(x, p=2.0, dim=1)
        x = self.cosine_layer(x)

        return x
