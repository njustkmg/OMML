import paddle
import paddle.nn as nn


class ContrastiveLoss(nn.Layer):
    """
    Compute contrastive loss
    """
    def __init__(self, margin=0, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, scores):
        # compute image-sentence score matrix
        diag_idx = [[i, i] for i in range(len(scores))]
        diagonal = paddle.gather_nd(scores, paddle.to_tensor(diag_idx)).unsqueeze(1)

        d1 = diagonal.expand_as(scores)
        d2 = paddle.transpose(d1, (1,0)).expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clip(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clip(min=0)

        # clear diagonals
        mask = paddle.eye(scores.shape[0]) < .5
        cost_s = cost_s * mask
        cost_im = cost_im * mask

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)
            cost_im = cost_im.max(0)
        return cost_s.sum() + cost_im.sum()