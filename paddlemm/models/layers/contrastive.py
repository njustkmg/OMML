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
        # print(scores)
        # compute image-sentence score matrix
        diagonal = paddle.reshape(paddle.diag(scores), [scores.shape[0], 1])
        d1 = diagonal.expand_as(scores).clone()
        d2 = diagonal.reshape([1, scores.shape[0]]).expand_as(scores)
        # d2 = diagonal.t().expand_as(scores).clone()

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clip(min=0)
        # cost_s = self.margin + scores - d1
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clip(min=0)
        # cost_im = self.margin + scores - d2
        # clear diagonals
        # cost_s = cost_s - paddle.diag(paddle.diag(cost_s))
        # cost_im = cost_im - paddle.diag(paddle.diag(cost_im))
        mask = (paddle.eye(scores.shape[0]) < .5)
        cost_s = cost_s * mask
        cost_im = cost_im * mask

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)
            cost_im = cost_im.max(0)
        return cost_s.sum() + cost_im.sum()