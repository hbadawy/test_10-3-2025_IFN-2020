
import torch
import torch.nn as nn

class cd_loss(nn.Module):
    def __init__(self):
        super(cd_loss, self).__init__()
        self.bce_loss = nn.BCELoss()
    def forward(self, input, target):
        bce_loss = self.bce_loss(torch.sigmoid(input), target)
        smooth = 1.
        iflat = input.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()
        dic_loss = 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))
        return dic_loss + bce_loss


# def cd_loss(input,target):
#     bce_loss = nn.BCELoss()
#     bce_loss = bce_loss(torch.sigmoid(input),target)

#     smooth = 1.
#     iflat = input.view(-1)
#     tflat = target.view(-1)
#     intersection = (iflat * tflat).sum()
#     dic_loss = 1 - ((2. * intersection + smooth)/(iflat.sum() + tflat.sum() + smooth))

#     return  dic_loss + bce_loss