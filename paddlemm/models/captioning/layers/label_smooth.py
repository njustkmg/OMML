import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class LabelSmoothing(nn.Layer):
    "Implement label smoothing."
    def __init__(self, vocab_size, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='none')
        self.vocab_size = vocab_size
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.shape[1]]
        mask = mask[:, :input.shape[1]]

        input = input.reshape((-1, input.shape[-1]))
        target = target.reshape((-1, ))
        mask = mask.reshape((-1, ))

        self.size = input.shape[1]
        target_one_hot = F.one_hot(target, num_classes=self.vocab_size)
        x = paddle.full(target_one_hot.shape, dtype=target_one_hot.dtype, fill_value=self.confidence)
        y = paddle.full(target_one_hot.shape, dtype=target_one_hot.dtype, fill_value=self.smoothing / (self.size - 1))
        true_dist = paddle.where(target_one_hot!=0, x, y)

        return (self.criterion(input, true_dist).sum(1) * mask).sum() / mask.sum()
