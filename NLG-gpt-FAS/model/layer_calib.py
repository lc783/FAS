# from cv2 import mean
# from sympy import print_rcode
import torch
import torch.nn as nn
from torch.nn import functional as F

class MergeTemporalDim(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T

    def forward(self, x_seq: torch.Tensor):
        return x_seq.flatten(0, 1).contiguous()

class ExpandTemporalDim(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T

    def forward(self, x_seq: torch.Tensor):
        y_shape = [self.T, int(x_seq.shape[0]/self.T)]
        y_shape.extend(x_seq.shape[1:])
        return x_seq.view(y_shape)

class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama):
        out = (input >= 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None

class GradFloor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.floor()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

myfloor = GradFloor.apply

class IF(nn.Module):
    def __init__(self, T=0, L=8, thresh=2.0, tau=1., gama=1.0, last=False, align=False):
        super(IF, self).__init__()
        self.act = ZIF.apply
        self.thresh = nn.Parameter(torch.tensor([thresh]), requires_grad=True)
        self.thresh2 = nn.Parameter(torch.full((1024, 3072), thresh), requires_grad=True)
        self.tau = tau
        self.gama = gama
        self.expand = ExpandTemporalDim(T)
        self.merge = MergeTemporalDim(T)
        self.L = L
        self.T = T
        self.loss = 0
        self.loss2 = 0

        self.dtmem = nn.Parameter(torch.ones_like(self.thresh2)*0.0, requires_grad=True)
        self.align = align

    def forward(self, x):
        if self.align:
            self.loss = 0

            batchSize = int(x.size()[0] / (self.T + 1))
            snn_data = x[:-batchSize]
            ANN_data = x[-batchSize:]

            thre = self.thresh2
            snn_data = self.expand(snn_data)

            mem = self.dtmem * thre

            spike_pot = []
            for t in range(self.T):
                mem = mem + snn_data[t, ...]
                spike = self.act(mem - thre, self.gama).type_as(thre)
                spike = spike * thre
                mem = mem - spike
                spike_pot.append(spike)


            snn_data = torch.stack(spike_pot, dim=0)

            snn_out = snn_data.mean(0)

            snn_data = self.merge(snn_data)

            ANN_data = ANN_data / self.thresh.data
            ANN_data = torch.clamp(ANN_data, 0, 1)
            ANN_data = myfloor(ANN_data * self.L + 0.5) / self.L
            ANN_data = ANN_data * self.thresh.data
            self.loss = F.mse_loss(snn_out, ANN_data.data)

            x = torch.cat((snn_data, ANN_data), dim=0)
            return x

        else:
            if self.T > 0:
                thre = self.thresh2.data
                x = self.expand(x)
                mem = self.dtmem * thre
                spike_pot = []
                for t in range(self.T):
                    mem = mem + x[t, ...]
                    spike = self.act(mem - thre, self.gama).type_as(thre)
                    spike = spike * thre
                    mem = mem - spike
                    spike_pot.append(spike)
                x = torch.stack(spike_pot, dim=0)
                x = self.merge(x)
            else:
                x = x / self.thresh
                x = torch.clamp(x, 0, 1)
                x = myfloor(x*self.L+0.5)/self.L
                x = x * self.thresh
        return x

    def set_thres2(self):
        with torch.no_grad():
            self.thresh2.data.fill_(self.thresh[0].data)

def add_dimention(x, T):
    x.unsqueeze_(1)
    x = x.repeat(T, 1, 1, 1)
    return x

def add_dimention_mask(x, T):
    x.unsqueeze_(1)
    x = x.repeat(T, 1, 1, 1, 1)
    return x
