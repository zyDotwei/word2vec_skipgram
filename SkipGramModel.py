import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class SkipGramModel(nn.Module):

    def __init__(self, vocab_size, emb_dim=100):
        super(SkipGramModel, self).__init__()

        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.v_embeddings = nn.Embedding(vocab_size, emb_dim, sparse=True)   # 中心词向量
        self.u_embeddings = nn.Embedding(vocab_size, emb_dim, sparse=True)   # 背景词向量
        self.init_emb()

    def init_emb(self):
        """Initialize embedding weight like word2vec.
        The u_embedding is a uniform distribution in [-0.5/em_size, 0.5/emb_size], and the elements of v_embedding are zeroes.
        """
        initrange = 0.5 / self.emb_dim
        self.v_embeddings.weight.data.uniform_(-initrange, initrange)
        self.u_embeddings.weight.data.uniform_(-1.0, 1.0)

    def forward(self, pos_v, pos_u, neg_u):
        """Forward process.
        As pytorch designed, all variables must be batch format, so all input of this method is a list of word id.
        Args:
            pos_v: list of center word ids for positive word pairs.
            pos_u: list of neibor word ids for positive word pairs.
            neg_u: list of center word ids for negative word pairs.

        Returns:
            Loss of this process, a pytorch variable.
        """
        emb_v = self.v_embeddings(pos_v)   # [batch, emb]
        emb_u = self.u_embeddings(pos_u)   # [batch, emb]
        # [batch, 1, emb] * # [batch, emb, 1]  ==> [batch]
        score = torch.bmm(emb_u.unsqueeze(1), emb_v.unsqueeze(2)).squeeze()
        score = F.logsigmoid(score)   # [batch]

        neg_emb_u = self.u_embeddings(neg_u)   # [batch, neg_eg, emb]
        # [batch, neg_eg, emb] * # [batch, emb, 1]  ==> [batch, neg_eg, 1] ==> [batch, neg_eg]
        neg_score = torch.bmm(neg_emb_u, emb_v.unsqueeze(2)).squeeze()
        neg_score = F.logsigmoid(-1 * neg_score)
        return -1*(torch.sum(score) + torch.sum(neg_score))

    def get_embedding_weight_u(self, use_cuda):
        if use_cuda:
            embedding = self.u_embeddings.weight.cpu().data.numpy()
        else:
            embedding = self.u_embeddings.weight.data.numpy()
        return embedding

    def get_embedding_weight_v(self, use_cuda):
        if use_cuda:
            embedding = self.v_embeddings.weight.cpu().data.numpy()
        else:
            embedding = self.v_embeddings.weight.data.numpy()
        return embedding


if __name__ == '__main__':
    model = SkipGramModel(50)

    pos_v = torch.LongTensor([2, 4])
    pos_u = torch.LongTensor([3, 5])
    neg_u = torch.LongTensor([[1, 6], [7, 8]])

    loss = model(pos_v, pos_u, neg_u)
    print(loss.data)
