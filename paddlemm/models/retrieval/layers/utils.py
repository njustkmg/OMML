import paddle


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = (paddle.abs(X).sum(axis=dim, keepdim=True) + eps).clone()
    X = paddle.divide(X, norm)
    return X

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = paddle.pow(X, 2).sum(axis=dim, keepdim=True).sqrt() + eps
    X = paddle.divide(X, norm)
    return X


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = paddle.sum(x1 * x2, dim)
    w1 = paddle.norm(x1, 2, dim)
    w2 = paddle.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clip(min=eps)).squeeze()


def cosine_similarity_a2a(x1, x2, dim=1, eps=1e-8):
    # x1: (B, n, d) x2: (B, m, d)
    w12 = paddle.bmm(x1, x2.transpose(1, 2))
    # w12: (B, n, m)

    w1 = paddle.norm(x1, 2, dim).unsqueeze(2)
    w2 = paddle.norm(x2, 2, dim).unsqueeze(1)

    # w1: (B, n, 1) w2: (B, 1, m)
    w12_norm = paddle.bmm(w1, w2).clip(min=eps)
    return w12 / w12_norm


def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score