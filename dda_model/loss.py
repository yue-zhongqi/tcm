import torch
import numpy as np
import torch.nn as nn


class NMTCritierion(nn.Module):
    def __init__(self, label_smoothing=0.0):
        super(NMTCritierion, self).__init__()
        self.label_smoothing = label_smoothing
        self.LogSoftmax = nn.LogSoftmax()
 
        if label_smoothing > 0:
            self.criterion = nn.KLDivLoss(size_average=False)
        else:
            self.criterion = nn.NLLLoss(size_average=False, ignore_index=100000)
        self.confidence = 1.0 - label_smoothing
 
    def _smooth_label(self, num_tokens):
        # When label smoothing is turned on,
        # KL-divergence between q_{smoothed ground truth prob.}(w)
        # and p_{prob. computed by model}(w) is minimized.
        # If label smoothing value is set to zero, the loss
        # is equivalent to NLLLoss or CrossEntropyLoss.
        # All non-true labels are uniformly set to low-confidence.
        one_hot = torch.randn(1, num_tokens)
        one_hot.fill_(self.label_smoothing / (num_tokens - 1))
        return one_hot
 
    def _bottle(self, v):
        return v.view(-1, v.size(2))
 
    def forward(self, dec_outs, labels):
        scores = self.LogSoftmax(dec_outs)
        num_tokens = scores.size(-1)
 
        # conduct label_smoothing module
        gtruth = labels.view(-1)
        if self.confidence < 1:
            tdata = gtruth.detach()
            one_hot = self._smooth_label(num_tokens)  # Do label smoothing, shape is [M]
            if labels.is_cuda:
                one_hot = one_hot.cuda()
            tmp_ = one_hot.repeat(gtruth.size(0), 1)  # [N, M]
            tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)  # after tdata.unsqueeze(1) , tdata shape is [N,1]
            gtruth = tmp_.detach()
        loss = self.criterion(scores, gtruth) / float(gtruth.shape[0])
        return loss


class AdversarialLoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(AdversarialLoss, self).__init__()
        self.epsilon = epsilon
        return
    
    def forward(self, input_, label, weight):
        entropy = - label * torch.log(input_ + self.epsilon) - (1 - label) * torch.log(1 - input_ + self.epsilon)
        return torch.sum(entropy * weight) / 2 

def Entropy(input_):
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy

def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()
    return fun1

def get_alignment_loss(x, bridge, ad_net, coeff, GVBD, softmax_logits, use_logits=True, no_entropy=False):
    ad_out, fc_out = ad_net(x)
    if GVBD > 0.0:
        ad_out = nn.Sigmoid()(ad_out - fc_out)
    else:
        ad_out = nn.Sigmoid()(ad_out)
    batch_size = x.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()

    if use_logits:
        softmax_logits = x
    entropy = Entropy(softmax_logits)
    entropy.register_hook(grl_hook(coeff))
    entropy = torch.exp(-entropy)
    mean_entropy = torch.mean(entropy)
    gvbd = torch.mean(torch.abs(fc_out))
    gvbg = torch.mean(torch.abs(bridge))

    source_mask = torch.ones_like(entropy)
    source_mask[x.size(0) // 2:] = 0
    source_weight = entropy * source_mask
    target_mask = torch.ones_like(entropy)
    target_mask[0:x.size(0) // 2] = 0
    target_weight = entropy * target_mask
    weight = source_weight / torch.sum(source_weight).detach().item() + \
             target_weight / torch.sum(target_weight).detach().item()
    if no_entropy:
        weight = torch.ones(weight.shape).to(weight.device) * 2.0 / float(weight.shape[0])
    loss = AdversarialLoss()
    return loss(ad_out, dc_target, weight.view(-1, 1)), mean_entropy, gvbd, gvbg

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)