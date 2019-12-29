from collections import defaultdict
import numpy as np
import random
import math
import os
from tensorboardX import SummaryWriter
from utils.build_dataset import BuildDataSet
import torch.utils.data as Data
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
from SkipGramModel import SkipGramModel
from tqdm import tqdm
import time
import logging


class MyWord2Vec:

    def __init__(self, use_cuda=False, outputdir=None, device_ids=[0]):

        self.writer = SummaryWriter()
        self.sentences_count = 0  # 保存句子数
        self.word_frequency = defaultdict(int)   # 词频统计
        self.total_words = 0    # 单词总数(不去重)
        self.retain_word_total = 0  # 去掉小于min_count词后剩余全部
        self.skipgram_dataset_count = 0
        self.sample_table = []      # 采样table

        self.word2id = defaultdict(int)
        self.id2word = defaultdict(int)
        # add UNK
        UNK = 'UNK'
        self.word2id[UNK] = 0
        self.id2word[0] = UNK

        self.embedding_weight_u = None
        self.embedding_weight_v = None
        if use_cuda:
            self.use_cuda = torch.cuda.is_available()
        self.device_ids = device_ids
        self.outputdir = outputdir
        if self.outputdir is not None:
            if not os.path.exists(self.outputdir):
                os.makedirs(self.outputdir)

    def word2vec(self, filename, emb_dim=100, window=2, min_count=1, batch_size=2048, iter=5):

        self.batch_size = batch_size
        self.raw_file = self._get_file_name(filename)
        self.samplingfile = filename
        self.skipgram_dataset = None

        self.build_vocab(filename, min_count, window)
        self.init_sample_table()
        self.model = SkipGramModel(len(self.word2id), emb_dim)
        if self.use_cuda:
            self.batch_size = self.batch_size * len(self.device_ids)
            self.model = nn.DataParallel(self.model, device_ids=self.device_ids)  # multi-GPU
            self.model.cuda(device=self.device_ids[0])
        self.train(batch_size, iter)

    def _get_file_name(self, filepath):
        _, filename = os.path.split(filepath)
        file_name, _ = os.path.splitext(filename)
        return file_name

    def read_data(self, filename):
        with open(filename, 'r', encoding='utf-8') as fr:
            for line in fr.readlines():
                yield line.strip()

    def _scan_vocab(self, sentences, min_count):

        total_sentences = 0
        total_words = 0  # 保存出现的单词总数（不去重)
        word_frequency = defaultdict(int)
        logging.info('开始处理语料...')
        for sentence_id, line in enumerate(sentences):
            sentence = line.split(' ')
            for word in sentence:
                word_frequency[word] += 1
            total_words += len(sentence)  # 记录总单词数
            total_sentences += 1

        word_id = 1
        for word, count in word_frequency.items():
            if count < min_count:
                continue
            self.word2id[word] = word_id
            self.id2word[word_id] = word
            self.word_frequency[word_id] = count
            self.retain_word_total += count
            word_id += 1

        self.sentences_count = total_sentences
        self.total_words = total_words
        logging.info('sentences_count:{}'.format(self.sentences_count))
        logging.info('total_words:{}'.format(self.total_words))
        logging.info('min_count:{}, retain_word_total:{}'.format(min_count, self.retain_word_total))
        logging.info('unique word:{}'.format(len(self.word2id)))
        logging.info('处理语料完毕.')

    def _sub_sampling(self, sentences, threshold=5e-5):

        if self.outputdir is not None:
            self.samplingfile = os.path.join(self.outputdir, self.raw_file+'_sampling.txt')
        else:
            self.samplingfile = self.raw_file+'_sampling.txt'
        logging.info('开始subSample...')
        fp = open(self.samplingfile, 'w+', encoding='utf-8')
        for id, line in enumerate(sentences):
            sentence = line.split(' ')
            cur_sentence = []
            for word in sentence:
                wid = self.word2id[word]
                if wid == 0:
                    continue
                sub_ram = math.sqrt(threshold/(self.word_frequency[wid]/self.retain_word_total))
                if random.random() < sub_ram:
                    cur_sentence.append(word)
            fp.write(' '.join(cur_sentence) + '\n')
        fp.close()
        logging.info('subSample完成.')

    def generate_skipgram_dataset(self, window):

        if self.outputdir is not None:
            self.skipgram_dataset = os.path.join(self.outputdir, self.raw_file + '_skipgram.txt')
        else:
            self.skipgram_dataset = self.raw_file + '_skipgram.txt'
        logging.info('开始建立skipgram_dataset...')
        fp = open(self.skipgram_dataset, 'w+', encoding='utf-8')
        for line in self.read_data(self.samplingfile):
            word_ids = []
            for word in line.strip().split(' '):
                word_ids.append(self.word2id[word])

            for i_id, pos_v in enumerate(word_ids):
                for pos_u in word_ids[max(i_id-window, 0):i_id+window+1]:
                    if pos_v == pos_u:
                        continue
                    fp.write(self.id2word[pos_v] + ' '+ str(pos_v) + ' ' + \
                             self.id2word[pos_u] + ' ' +str(pos_u) + '\n')
                    self.skipgram_dataset_count += 1
        fp.close()
        logging.info('skipgram_dataset_count:{}'.format(self.skipgram_dataset_count))
        logging.info('建立skipgram_dataset完成.')

    def init_sample_table(self, table_size=1e8):

        logging.info('开始建立sample_table...')
        pow_frequency = np.array(list(self.word_frequency.values())) ** 0.75
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = np.round(ratio * table_size)

        for word_id, ct in zip(self.word_frequency.keys(), count):
            self.sample_table += [word_id] * int(ct)
        self.sample_table = np.array(self.sample_table)
        logging.info('建立sample_table完成, 长度{}'.format(len(self.sample_table )))

    def get_neg_u_neg_sampling(self, positive_batch, count):

        neg_u = []
        for positive in positive_batch:
            while True:
                cur_neg_u = list(np.random.choice(self.sample_table, count))
                if positive not in cur_neg_u:
                    neg_u.append(cur_neg_u)
                    break
        return neg_u

    def build_vocab(self, path, min_count, window):
        self._scan_vocab(self.read_data(path), min_count)
        self._sub_sampling(self.read_data(path))
        self.generate_skipgram_dataset(window)

    def train(self, batch_size, iter):

        train_data = BuildDataSet(self.skipgram_dataset, self.skipgram_dataset_count)
        data_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, drop_last=False)

        optimizer = torch.optim.SparseAdam(self.model.parameters(), lr=0.005)

        flag = False
        total_batch = 0
        last_improve = 0
        best_loss = float('inf')
        for epoch in range(iter):
            epoch_loss = []
            logging.info('Epoch [{}/{}]'.format(epoch + 1, iter))
            for i, (pos_v, pos_u) in enumerate(data_loader):
                neg_u = self.get_neg_u_neg_sampling(pos_v, 5)
                pos_v = Variable(torch.LongTensor(pos_v))
                pos_u = Variable(torch.LongTensor(pos_u))
                neg_u = Variable(torch.LongTensor(neg_u))
                if self.use_cuda:
                    pos_v = pos_v.cuda(device=self.device_ids[0])
                    pos_u = pos_u.cuda(device=self.device_ids[0])
                    neg_u = neg_u.cuda(device=self.device_ids[0])
                optimizer.zero_grad()
                if self.use_cuda:
                    loss = self.model(pos_v, pos_u, neg_u)[0]
                else:
                    loss = self.model(pos_v, pos_u, neg_u)
                loss.backward()
                optimizer.step()

                if loss < best_loss:
                    best_loss = loss.item()
                    last_improve = total_batch

                if total_batch % 50 == 0:
                    logging.info('epoch:{0}, barch：{1}, Loss: {2:>.8}'.format(epoch, total_batch, loss.item()))
                    self.writer.add_scalar('batch/loss', loss.item(), total_batch)
                #if total_batch - last_improve > 20000:
                #     # 验证集loss超过50000batch没下降，结束训练
                #    logging.info("No optimization for a long time, {}".format(total_batch - last_improve))
                #     flag = True
                #     break
                total_batch += 1
                epoch_loss.append(loss.item())
            # if flag:
            #     break
            self.writer.add_scalar('epoch/loss', np.mean(epoch_loss), epoch)
        self.writer.close()
        logging.info('模型训练完毕.')
        if isinstance(self.model, torch.nn.DataParallel):
            self.model = self.model.module
        self.embedding_weight_u = self.model.get_embedding_weight_u(self.use_cuda)
        self.embedding_weight_v = self.model.get_embedding_weight_v(self.use_cuda)

    def similarity(self, word1, word2):
        id_1 = self.word2id[word1]
        id_2 = self.word2id[word2]

        if id_1 == 0:
            raise Exception('错误，没有:{}'.format(word1))
        if id_2 == 0:
            raise Exception('错误，没有:{}'.format(word2))
        vec1 = self.embedding_weight_v[id_1]
        vec2 = self.embedding_weight_v[id_2]

        return (np.dot(vec1, vec2) / ((np.linalg.norm(vec1) * np.linalg.norm(vec2))))

    def most_similarity(self, word, nums=10):

        q_id = self.word2id[word]
        if q_id == 0:
            raise Exception('错误，没有:{}'.format(word))

        word_sim = []
        for id, word_q in self.id2word.items():
            sim = 0
            if id == q_id or id == 0:
                sim = 0
            else:
                try:
                    sim = self.similarity(word, word_q)
                except Exception as e:
                    print(e)
                    sim = 0
            word_sim.append(sim)
        word_sim = np.array(word_sim)
        word_idx = np.argsort(-word_sim)
        m_sim = dict()
        for idx in word_idx[:nums]:
            if word_sim[idx] > 0:
                m_sim[self.id2word[idx]] = word_sim[idx]
        return m_sim

    def save_embedding_u(self, file_name):

        fr = open(file_name, 'w', encoding='utf-8')
        for wid, w in self.id2word.items():
            e = self.embedding_weight_u[wid]
            e = ' '.join(map(lambda x: str(x), e))
            fr.write('%s %s\n' % (w, e))
        fr.close()

    def save_embedding_v(self, file_name):

        fr = open(file_name, 'w', encoding='utf-8')
        for wid, w in self.id2word.items():
            e = self.embedding_weight_v[wid]
            e = ' '.join(map(lambda x: str(x), e))
            fr.write('%s %s\n' % (w, e))
        fr.close()

    def load(self, path):
        embedding_weight = []
        wid = 0
        with open(path, 'r', encoding='utf-8') as fr:
            for line in tqdm(fr.readlines()):
                embedding = line.strip().split()
                self.word2id[embedding[0]] = wid
                self.id2word[wid] = embedding[0]

                cur_weight = [float(weight) for weight in embedding[1:]]
                embedding_weight.append(cur_weight)
                wid += 1
            self.embedding_weight_v = np.array(embedding_weight)
        return self

    def get_embedding(self, word):
        id_ = self.word2id[word]
        if id_ == 0:
            raise Exception('错误，没有:{}'.format(word))
        return self.embedding_weight_v[id_]

    def embedding_visualization(self):
        self.writer.add_embedding(self.embedding_weight_v, list(self.id2word.values()))
        self.writer.close()


def load_model_simtest(model, filename, output=None):

    if output is not None:
        fw = open(output, 'w+', encoding='utf-8')

    with open(filename, 'r', encoding='utf-8') as fr:
        for line in tqdm(fr.readlines()):
            line = line.strip()
            word1, word2 = line.split('\t')
            try:
                sim = model.similarity(word1, word2)
            except Exception as e:
                print(e)
                sim = 'oov'
            finally:
                wordstr = word1 + '\t' + word2 + '\t' + str(sim)
                if output is not None:
                    fw.write(wordstr + '\n')
                else:
                    logging.info(wordstr)

    if output is not None:
        fw.close()


if __name__ == '__main__':

    output_dir = 'output'

    logging.basicConfig(filename=os.path.join(output_dir, 'logging.log'),
                        filemode="w",
                        format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    model = MyWord2Vec(use_cuda=True, outputdir=output_dir, device_ids=[3, 1])
    model.word2vec('./data/wiki_zh_mini.txt', emb_dim=100, window=2, min_count=1, batch_size=2048, iter=8)
    model.save_embedding_u(os.path.join(output_dir, 'model_u.txt'))
    model.save_embedding_v(os.path.join(output_dir, 'model_v.txt'))

    model = model.load(path=os.path.join(output_dir, 'model_v.txt'))
    load_model_simtest(model, 'pku_sim_test.txt', os.path.join(output_dir, 'pku_sim_result_self.txt'))
