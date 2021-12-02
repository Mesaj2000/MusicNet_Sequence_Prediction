import torch


EPSILON = 0.0000000001

"""
input: a and b are pytorch tensors
       must be the same shape
output: ratio of identical element-wise pairs, NOT including (0, 0)
"""
def jaccard_stupid(a, b, device):
    zero = torch.zeros(a.shape).to(device)
    a_is_zero = a == zero
    b_is_zero = b == zero
    both_zero = torch.logical_and(a_is_zero, b_is_zero)
    num_matches = (a == b).sum()
    num_both_zero = both_zero.sum()
    num_nonzero_matches = torch.sub(num_matches, num_both_zero)
    total_nonzero = torch.sub((a == a).sum(), num_both_zero)

    if total_nonzero.item() == 0:
        return torch.div(num_matches, num_matches)

    return torch.div(num_nonzero_matches, total_nonzero)


def jaccard(a, b):
    intersection = torch.logical_and(a, b)
    union = torch.logical_or(a, b)
    IoU = intersection.sum().item() / union.sum().item()
    return IoU


# Doesn't work
def jaccard_loss_original(a, b, device):
    ratio = jaccard(a, b, device, printout_stuff=True)
    ratio = torch.add(ratio, EPSILON)
    loss = torch.mul(-1, torch.log(ratio))
    return loss


# credit: https://gist.github.com/wassname/
def jaccard_loss(y_pred, y_true, smooth=100):
    intersection = (y_true * y_pred).abs().sum(dim=-1)
    sum_ = torch.sum(y_true.abs() + y_pred.abs(), dim=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    output = (1 - jac) * smooth
    return output.mean()

# credit: https://gist.github.com/wassname/
class JaccardDistanceLoss(torch.nn.Module):    

    def __init__(self, smooth=100, dim=1, size_average=True, reduce=True):
        """
        Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
                = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
        The jaccard distance loss is usefull for unbalanced datasets. This has been
        shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
        gradient.
        Ref: https://en.wikipedia.org/wiki/Jaccard_index
        @url: https://gist.github.com/wassname/d1551adac83931133f6a84c5095ea101
        @author: wassname
        """
        super(JaccardDistanceLoss, self).__init__()
        self.smooth = smooth
        self.dim = dim
        self.size_average = size_average
        self.reduce = reduce

    def forward(self, y_pred, y_true):
        intersection = (y_true * y_pred).abs().sum(self.dim)
        sum_ = (y_true.abs() + y_pred.abs()).sum(self.dim)
        jac = (intersection + self.smooth) / (sum_ - intersection + self.smooth)
        losses = (1 - jac) * self.smooth    
        if self.reduce:
            return losses.mean() if self.size_average else losses.sum()
        else:
            return losses