import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddlemm.models.fusion.backbone import Vgg, Resnet, Lstm, Gru


ModelMap = {
    'vgg': Vgg,
    'resnet': Resnet,
    'lstm': Lstm,
    'gru': Gru
}

# loss function
def KL(alpha, c):
    beta = paddle.ones((1, c))
    S_alpha = paddle.sum(alpha, axis=1, keepdim=True)
    S_beta = paddle.sum(beta, axis=1, keepdim=True)
    lnB = paddle.lgamma(S_alpha) - paddle.sum(paddle.lgamma(alpha), axis=1, keepdim=True)
    lnB_uni = paddle.sum(paddle.lgamma(beta), axis=1, keepdim=True) - paddle.lgamma(S_beta)
    dg0 = paddle.digamma(S_alpha)
    dg1 = paddle.digamma(alpha)
    kl = paddle.sum((alpha - beta) * (dg1 - dg0), axis=1, keepdim=True) + lnB + lnB_uni
    return kl

def ce_loss(p, alpha, c, global_step, annealing_step):
    S = paddle.sum(alpha, axis=1, keepdim=True)
    E = alpha - 1
    A = paddle.sum(p * (paddle.digamma(S) - paddle.digamma(alpha)), axis=1, keepdim=True)

    annealing_coef = min(1, global_step / annealing_step)

    alp = E * (1 - p) + 1
    B = annealing_coef * KL(alp, c)

    return (A + B)

def mse_loss(p, alpha, c, global_step, annealing_step=1):
    S = paddle.sum(alpha, axis=1, keepdim=True)
    E = alpha - 1
    m = alpha / S
    label = F.one_hot(p, num_classes=c)
    A = paddle.sum((label - m) ** 2, axis=1, keepdim=True)
    B = paddle.sum(alpha * (S - alpha) / (S * S * (S + 1)), axis=1, keepdim=True)
    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    C = annealing_coef * KL(alp, c)
    return (A + B) + C


class TMCFusion(nn.Layer):
    def __init__(self,
                 image_model,
                 text_model,
                 word_dim,
                 vocab_size,
                 hidden_dim,
                 num_labels,
                 lambda_epochs,
                 model_name,
                 finetune,
                 **kwargs):
        super(TMCFusion, self).__init__()

        self.num_labels = num_labels
        self.lambda_epochs = lambda_epochs
        self.image_model = ModelMap[image_model.lower()](hidden_dim, num_labels, model_name, finetune)
        self.text_model = ModelMap[text_model.lower()](word_dim, hidden_dim, num_labels, vocab_size, model_name)

        self.sigmoid = nn.Sigmoid()

    def DS_Combin(self, alpha):
        """
        :param alpha: All Dirichlet distribution parameters.
        :return: Combined Dirichlet distribution parameters.
        """
        def DS_Combin_two(alpha1, alpha2):
            """
            :param alpha1: Dirichlet distribution parameters of view 1
            :param alpha2: Dirichlet distribution parameters of view 2
            :return: Combined Dirichlet distribution parameters
            """
            alpha = dict()
            alpha[0], alpha[1] = alpha1, alpha2
            b, S, E, u = dict(), dict(), dict(), dict()
            for v in range(2):
                S[v] = paddle.sum(alpha[v], axis=1, keepdim=True)
                E[v] = alpha[v]-1
                b[v] = E[v]/(S[v].expand(E[v].shape))
                u[v] = self.num_labels/S[v]

            # b^0 @ b^(0+1)
            bb = paddle.bmm(b[0].reshape([-1, self.num_labels, 1]), b[1].reshape([-1, 1, self.num_labels]))
            # b^0 * u^1
            uv1_expand = u[1].expand(b[0].shape)
            bu = paddle.multiply(b[0], uv1_expand)
            # b^1 * u^0
            uv_expand = u[0].expand(b[0].shape)
            ub = paddle.multiply(b[1], uv_expand)
            # calculate C
            bb_sum = paddle.sum(bb, axis=(1, 2)) # [200,10,10]
            bb_diag = paddle.diagonal(bb, axis1=-2, axis2=-1).sum(-1)
            C = bb_sum - bb_diag

            # calculate b^a
            b_a = (paddle.multiply(b[0], b[1]) + bu + ub)/((1-C).reshape([-1, 1]).expand(b[0].shape))
            # calculate u^a
            u_a = paddle.multiply(u[0], u[1])/((1-C).reshape([-1, 1]).expand(u[0].shape))

            # calculate new S
            S_a = self.num_labels / u_a
            # calculate new e_k
            e_a = paddle.multiply(b_a, S_a.expand(b_a.shape))
            alpha_a = e_a + 1
            return alpha_a

        for v in range(len(alpha)-1):
            if v == 0:
                alpha_a = DS_Combin_two(alpha[0], alpha[1])
            else:
                alpha_a = DS_Combin_two(alpha_a, alpha[v+1])
        return alpha_a

    def forward(self, batch):

        img_hidden = self.image_model(batch)
        txt_hidden = self.text_model(batch)
        evidence = dict()
        evidence[0] = img_hidden
        evidence[1] = txt_hidden

        loss = 0
        alpha = dict()
        for i in range(2):
            alpha[i] = evidence[i] + 1
            loss += ce_loss(batch['label'], alpha[i], self.num_labels, batch['epoch'], self.lambda_epochs)
        alpha_a = self.DS_Combin(alpha)
        evidence_a = alpha_a - 1
        loss += ce_loss(batch['label'], alpha_a, self.num_labels, batch['epoch'], self.lambda_epochs)
        loss = paddle.mean(loss)
        predict = self.sigmoid(evidence_a)

        if self.training:
            return loss
        else:
            return loss, predict
