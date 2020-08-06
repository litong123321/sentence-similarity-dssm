# /usr/bin/env python
# coding=utf-8

indexes = []

import time

start_time = time.time()
import multiprocessing
import os
import re
import json
import gensim
import jieba
import keras
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
#print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
import keras.backend as K
import numpy as np
import pandas as pd
from itertools import combinations
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
from keras.activations import softmax
from keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback, Callback, ReduceLROnPlateau, \
    LearningRateScheduler
from keras.layers import LSTM
from keras.models import Model
from keras.optimizers import SGD, Adadelta, Adam, Nadam, RMSprop
from keras.regularizers import L1L2, l2
from keras.preprocessing.sequence import pad_sequences
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier

from gensim.models.word2vec import LineSentence
from gensim.models.fasttext import FastText
import copy

model_dir = '/home/litong/code/utils/'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

#####################################################################
#                         数据加载预处理阶段
#####################################################################
new_words = "支付宝 付款码 二维码 收钱码 转账 退款 退钱 余额宝 运费险 还钱 还款 花呗 借呗 蚂蚁花呗 蚂蚁借呗 " \
            "蚂蚁森林 小黄车 飞猪 微客 宝卡 芝麻信用 亲密付 淘票票 饿了么 摩拜 滴滴 滴滴出行".split(" ")
for word in new_words:
    jieba.add_word(word)

star = re.compile("\*+")

test_size = 0.025
random_state = 42
fast_mode, fast_rate = False, 0.01  # 快速调试，其评分不作为参考
train_file = model_dir + "atec_nlp_sim_train.csv"


def load_data(dtype="both", input_length=[20, 24], w2v_length=300):
    def __load_data(dtype="word", input_length=20, w2v_length=300):

        filename = model_dir + "%s_%d_%d" % (dtype, input_length, w2v_length)
        if os.path.exists(filename):
            return pd.read_pickle(filename)

        data_l_n = []
        data_r_n = []
        y = []
        for line in open(train_file, "r", encoding="utf8"):
            lineno, s1, s2, label = line.strip().split("\t")
            if dtype == "word":
                data_l_n.append([word2index[word] for word in list(jieba.cut(star.sub("1", s1))) if word in word2index])
                data_r_n.append([word2index[word] for word in list(jieba.cut(star.sub("1", s2))) if word in word2index])
            if dtype == "char":
                data_l_n.append([char2index[char] for char in s1 if char in char2index])
                data_r_n.append([char2index[char] for char in s2 if char in char2index])

            y.append(int(label))

        # 对齐语料中句子的长度
        data_l_n = pad_sequences(data_l_n, maxlen=input_length)
        data_r_n = pad_sequences(data_r_n, maxlen=input_length)
        y = np.array(y)

        pd.to_pickle((data_l_n, data_r_n, y), filename)

        return (data_l_n, data_r_n, y)

    if dtype == "both":
        ret_array = []
        for dtype, input_length in zip(['word', 'char'], input_length):
            data_l_n, data_r_n, y = __load_data(dtype, input_length, w2v_length)
            ret_array.append(np.asarray(data_l_n))
            ret_array.append(np.asarray(data_r_n))
        ret_array.append(y)
        return ret_array
    else:
        return __load_data(dtype, input_length, w2v_length)


def input_data(sent1, sent2, dtype="both", input_length=[20, 24]):
    def __input_data(sent1, sent2, dtype="word", input_length=20):
        data_l_n = []
        data_r_n = []
        for s1, s2 in zip(sent1, sent2):
            if dtype == "word":
                data_l_n.append([word2index[word] for word in list(jieba.cut(star.sub("1", s1))) if word in word2index])
                data_r_n.append([word2index[word] for word in list(jieba.cut(star.sub("1", s2))) if word in word2index])
            if dtype == "char":
                data_l_n.append([char2index[char] for char in s1 if char in char2index])
                data_r_n.append([char2index[char] for char in s2 if char in char2index])

        # 对齐语料中句子的长度
        data_l_n = pad_sequences(data_l_n, maxlen=input_length)
        data_r_n = pad_sequences(data_r_n, maxlen=input_length)

        return [data_l_n, data_r_n]

    if dtype == "both":
        ret_array = []
        for dtype, input_length in zip(['word', 'char'], input_length):
            data_l_n, data_r_n = __input_data(sent1, sent2, dtype, input_length)
            ret_array.append(data_l_n)
            ret_array.append(data_r_n)
        return ret_array
    else:
        return __input_data(sent1, sent2, dtype, input_length)


###########################################################################
#                            训练验证集划分
###########################################################################
def split_data(data, mode="train", test_size=test_size, random_state=random_state):
    # mode == "train":  划分成用于训练的四元组
    # mode == "orig":   划分成两组数据
    train = []
    test = []
    for data_i in data:
        if fast_mode:
            data_i, _ = train_test_split(data_i, test_size=1 - fast_rate, random_state=random_state)
        train_data, test_data = train_test_split(data_i, test_size=test_size, random_state=random_state)
        train.append(np.asarray(train_data))
        test.append(np.asarray(test_data))

    if mode == "orig":
        return train, test

    train_x, train_y, test_x, test_y = train[:-1], train[-1], test[:-1], test[-1]
    return train_x, train_y, test_x, test_y


#####################################################################
#                         模型定义
#####################################################################

w2v_length = 256
ebed_type = "gensim"
# ebed_type = "fastcbow"

if ebed_type == "gensim":
    char_embedding_model = gensim.models.Word2Vec.load(model_dir + "char2vec_gensim%s" % w2v_length)
    char2index = {v: k for k, v in enumerate(char_embedding_model.wv.index2word)}
    word_embedding_model = gensim.models.Word2Vec.load(model_dir + "word2vec_gensim%s" % w2v_length)
    word2index = {v: k for k, v in enumerate(word_embedding_model.wv.index2word)}


print("loaded w2v done!", len(char2index), len(word2index))

MAX_LEN = 30
MAX_EPOCH = 90
train_batch_size = 32
test_batch_size = 32
earlystop_patience, plateau_patience = 10, 10  # patience



def get_embedding_layers(dtype, input_length, w2v_length, with_weight=True):
    def __get_embedding_layers(dtype, input_length, w2v_length, with_weight=True):

        if dtype == 'word':
            embedding_length = len(word2index)
        elif dtype == 'char':
            embedding_length = len(char2index)

        if with_weight:
            if ebed_type == "gensim":
                if dtype == 'word':
                    embedding = word_embedding_model.wv.get_keras_embedding(train_embeddings=False)
                else:
                    embedding = char_embedding_model.wv.get_keras_embedding(train_embeddings=False)


        else:
            embedding = Embedding(embedding_length, w2v_length, input_length=input_length, trainable=True)

        return embedding

    if dtype == "both":
        embedding = []
        for dtype, input_length in zip(['word', 'char'], input_length):
            embedding.append(__get_embedding_layers(dtype, input_length, w2v_length, with_weight))
        return embedding
    else:
        return __get_embedding_layers(dtype, input_length, w2v_length, with_weight)


def create_pretrained_embedding(pretrained_weights_path, trainable=False, **kwargs):
    "Create embedding layer from a pretrained weights array"
    pretrained_weights = np.load(pretrained_weights_path)
    in_dim, out_dim = pretrained_weights.shape
    embedding = Embedding(in_dim, out_dim, weights=[pretrained_weights], trainable=False, **kwargs)
    return embedding


def unchanged_shape(input_shape):
    "Function for Lambda layer"
    return input_shape


def substract(input_1, input_2):
    "Substract element-wise"
    neg_input_2 = Lambda(lambda x: -x, output_shape=unchanged_shape)(input_2)
    out_ = Add()([input_1, neg_input_2])
    return out_


def submult(input_1, input_2):
    "Get multiplication and subtraction then concatenate results"
    mult = Multiply()([input_1, input_2])
    sub = substract(input_1, input_2)
    out_ = Concatenate()([sub, mult])
    return out_


def apply_multiple(input_, layers):
    "Apply layers to input then concatenate result"
    if not len(layers) > 1:
        raise ValueError('Layers list should contain more than 1 layer')
    else:
        agg_ = []
        for layer in layers:
            agg_.append(layer(input_))
        out_ = Concatenate()(agg_)
    return out_


def time_distributed(input_, layers):
    "Apply a list of layers in TimeDistributed mode"
    out_ = []
    node_ = input_
    for layer_ in layers:
        node_ = TimeDistributed(layer_)(node_)
    out_ = node_
    return out_


def soft_attention_alignment(input_1, input_2):
    "Align text representation with neural soft attention"
    attention = Dot(axes=-1)([input_1, input_2])
    w_att_1 = Lambda(lambda x: softmax(x, axis=1),
                     output_shape=unchanged_shape)(attention)
    w_att_2 = Permute((2, 1))(Lambda(lambda x: softmax(x, axis=2),
                                     output_shape=unchanged_shape)(attention))
    in1_aligned = Dot(axes=1)([w_att_1, input_1])
    in2_aligned = Dot(axes=1)([w_att_2, input_2])
    return in1_aligned, in2_aligned




class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
        self.supports_masking = True
        # self.init = initializations.get('glorot_uniform')
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='%s_W' % self.name,
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zero',
                                     name='%s_b' % self.name,
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        # eij = K.dot(x, self.W) TF backend doesn't support it

        # features_dim = self.W.shape[0]
        # step_dim = x._keras_shape[1]

        features_dim = self.features_dim
        step_dim = self.step_dim
        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)
        a = K.exp(eij)
        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        # print weigthted_input.shape
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        # return input_shape[0], input_shape[-1]
        return input_shape[0], self.features_dim


from keras.layers import Bidirectional, Dense, LSTM, Concatenate, \
    Multiply, Lambda, Dropout, Input, Maximum, Subtract,embeddings

def DSSM(pretrained_embedding, input_length, lstmsize=64):
    '''
    input1 input2 分别代表编码后的问题1、问题2的词特征

    input1c input2c 分别代表编码后的问题1、问题2的字特征

    input3 代表两个问题重合的词组

    '''
    word_embedding = pretrained_embedding
    wordlen = input_length

    input1 = Input(shape=(wordlen,))
    input2 = Input(shape=(wordlen,))
    lstm0 = LSTM(lstmsize, return_sequences=True)
    lstm1 = Bidirectional(LSTM(lstmsize))
    lstm2 = LSTM(lstmsize)
    #att1 = Attention(wordlen)
    den = Dense(64, activation='tanh')

    # att1 = Lambda(lambda x: K.max(x,axis = 1))

    v1 = word_embedding(input1)
    v2 = word_embedding(input2)
    v11 = lstm1(v1)
    v22 = lstm1(v2)
    v1ls = lstm2(lstm0(v1))
    v2ls = lstm2(lstm0(v2))
    v1=v11
    v2=v22
    #v1 = Concatenate(axis=1)([att1(v1), v11])
    #v2 = Concatenate(axis=1)([att1(v2), v22])

    #input1c = Input(shape=(charlen,))
    #input2c = Input(shape=(charlen,))
    #lstm1c = Bidirectional(LSTM(lstmsize))
    #att1c = Attention(charlen)
    #v1c = char_embedding(input1c)
    #v2c = char_embedding(input2c)
    #v11c = lstm1c(v1c)
    #v22c = lstm1c(v2c)
    #v1c=v11c
    #v2c=v22c
    #v1c = Concatenate(axis=1)([att1c(v1c), v11c])
    #v2c = Concatenate(axis=1)([att1c(v2c), v22c])

    mul = Multiply()([v1, v2])
    sub = Lambda(lambda x: K.abs(x))(Subtract()([v1, v2]))
    maximum = Maximum()([Multiply()([v1, v1]), Multiply()([v2, v2])])
    #mulc = Multiply()([v1c, v2c])
    #subc = Lambda(lambda x: K.abs(x))(Subtract()([v1c, v2c]))
    #maximumc = Maximum()([Multiply()([v1c, v1c]), Multiply()([v2c, v2c])])
    sub2 = Lambda(lambda x: K.abs(x))(Subtract()([v1ls, v2ls]))
    matchlist = Concatenate(axis=1)([mul, sub,maximum, sub2])
    matchlist = Dropout(0.05)(matchlist)

    matchlist = Concatenate(axis=1)(
        [Dense(32, activation='relu')(matchlist), Dense(48, activation='sigmoid')(matchlist)])
    res = Dense(1, activation='sigmoid')(matchlist)

    model = Model(inputs=[input1, input2], outputs=res)
    return model


"""
    From the paper:
        Averaging Weights Leads to Wider Optima and Better Generalization
        Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry Vetrov, Andrew Gordon Wilson
        https://arxiv.org/abs/1803.05407
        2018

    Author's implementation: https://github.com/timgaripov/swa
"""


class SWA(Callback):
    def __init__(self, model, swa_model, swa_start):
        super().__init__()
        self.model, self.swa_model, self.swa_start = model, swa_model, swa_start

    def on_train_begin(self, logs=None):
        self.epoch = 0
        self.swa_n = 0

    def on_epoch_end(self, epoch, logs=None):
        if (self.epoch + 1) >= self.swa_start:
            self.update_average_model()
            self.swa_n += 1

        self.epoch += 1

    def update_average_model(self):
        # update running average of parameters
        alpha = 1. / (self.swa_n + 1)
        for layer, swa_layer in zip(self.model.layers, self.swa_model.layers):
            weights = []
            for w1, w2 in zip(swa_layer.get_weights(), layer.get_weights()):
                weights.append((1 - alpha) * w1 + alpha * w2)
            swa_layer.set_weights(weights)


class LR_Updater(Callback):
    '''
    Abstract class where all Learning Rate updaters inherit from. (e.g., CircularLR)
    Calculates and updates new learning rate and momentum at the end of each batch.
    Have to be extended.
    '''

    def __init__(self, init_lrs):
        self.init_lrs = init_lrs

    def on_train_begin(self, logs=None):
        self.update_lr()

    def on_batch_end(self, batch, logs=None):
        self.update_lr()

    def update_lr(self):
        # cur_lrs = K.get_value(self.model.optimizer.lr)
        new_lrs = self.calc_lr(self.init_lrs)
        K.set_value(self.model.optimizer.lr, new_lrs)

    def calc_lr(self, init_lrs): raise NotImplementedError


class CircularLR(LR_Updater):
    '''
    A learning rate updater that implements the CircularLearningRate (CLR) scheme.
    Learning rate is increased then decreased linearly.
    '''

    def __init__(self, init_lrs, nb, div=4, cut_div=8, on_cycle_end=None):
        self.nb, self.div, self.cut_div, self.on_cycle_end = nb, div, cut_div, on_cycle_end
        super().__init__(init_lrs)

    def on_train_begin(self, logs=None):
        self.cycle_iter, self.cycle_count = 0, 0
        super().on_train_begin()

    def calc_lr(self, init_lrs):
        cut_pt = self.nb // self.cut_div
        if self.cycle_iter > cut_pt:
            pct = 1 - (self.cycle_iter - cut_pt) / (self.nb - cut_pt)
        else:
            pct = self.cycle_iter / cut_pt
        res = init_lrs * (1 + pct * (self.div - 1)) / self.div
        self.cycle_iter += 1
        if self.cycle_iter == self.nb:
            self.cycle_iter = 0
            if self.on_cycle_end: self.on_cycle_end(self, self.cycle_count)
            self.cycle_count += 1
        return res


class TimerStop(Callback):
    """docstring for TimerStop"""

    def __init__(self, start_time, total_seconds):
        super(TimerStop, self).__init__()
        self.start_time = start_time
        self.total_seconds = total_seconds
        self.epoch_seconds = []

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_seconds.append(time.time() - self.epoch_start)

        mean_epoch_seconds = sum(self.epoch_seconds) / len(self.epoch_seconds)
        if time.time() + mean_epoch_seconds > self.start_time + self.total_seconds:
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        print('timer stopping')


# ("dssm",  "both", [20,24], ebed_type,  w2v_length, [],           124-8, earlystop_patience)
def get_model(cfg, model_weights=None):
    print("=======   CONFIG ======: ", cfg)

    model_type, dtype, input_length, ebed_type, w2v_length, n_hidden, n_epoch, patience = cfg
    embedding = get_embedding_layers(dtype, input_length, w2v_length, with_weight=True)


    if model_type == "dssm":
        model = DSSM(pretrained_embedding=embedding, input_length=input_length, lstmsize=64)

    if model_weights is not None:
        model.load_weights(model_weights)

    # keras.utils.plot_model(model, to_file=model_dir+model_type+"_"+dtype+'.png', show_shapes=True, show_layer_names=True, rankdir='TB')
    return model


#####################################################################
#                         评估指标和最佳阈值
#####################################################################

def r_f1_thresh(y_pred, y_true, step=1000):
    e = np.zeros((len(y_true), 2))
    e[:, 0] = y_pred.reshape(-1)
    e[:, 1] = y_true
    f = pd.DataFrame(e)
    thrs = np.linspace(0, 1, step + 1)
    x = np.array([f1_score(y_pred=f.loc[:, 0] > thr, y_true=f.loc[:, 1]) for thr in thrs])
    f1_, thresh = max(x), thrs[x.argmax()]
    return f.corr()[0][1], f1_, thresh


#####################################################################
#                         模型训练和保存
#####################################################################
configs_path = model_dir + "all_configs.json"


def save_config(filepath, cfg):
    configs = {}
    if os.path.exists(configs_path): configs = json.loads(open(configs_path, "r", encoding="utf8").read())
    configs[filepath] = cfg
    open(configs_path, "w", encoding="utf8").write(json.dumps(configs, indent=2, ensure_ascii=False))


def train_model(model, swa_model, cfg):
    model_type, dtype, input_length, ebed_type, w2v_length, n_hidden, n_epoch, patience = \
        ("dssm", "both", [20, 24], gensim , 256 , [], 20 , earlystop_patience)

    data = load_data(dtype, input_length, w2v_length)
    train_x, train_y, test_x, test_y = split_data(data)
    filepath = model_dir + model_type + "_" + dtype + time.strftime("_%m-%d %H-%M-%S") + ".h5"  # 每次运行的模型都进行保存，不覆盖之前的结果
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True,
                                 mode='auto')
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=0, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', verbose=0, factor=0.5, patience=2, min_lr=1e-6)
    swa_cbk = SWA(model, swa_model, swa_start=1)

    init_lrs = 0.001
    clr_div, cut_div = 10, 8
    batch_num = (train_x[0].shape[0] - 1) // train_batch_size + 1
    cycle_len = 1
    total_iterators = batch_num * cycle_len
    print("total iters per cycle(epoch):", total_iterators)
    #circular_lr = CircularLR(init_lrs, total_iterators, on_cycle_end=None, div=clr_div, cut_div=cut_div)
    #callbacks = [checkpoint, earlystop, swa_cbk, circular_lr]
    #callbacks.append(TimerStop(start_time=start_time, total_seconds=7100))

    def fit(n_epoch=n_epoch):
        history = model.fit(x=train_x, y=train_y,
                            #class_weight={0: 1 / np.mean(train_y), 1: 1 / (1 - np.mean(train_y))},
                            validation_data=((test_x, test_y)),
                            batch_size=train_batch_size,
                            #callbacks=callbacks,
                            epochs=n_epoch , verbose=2)
        return history

    loss, metrics = 'binary_crossentropy', ['binary_crossentropy', "accuracy"]

    model.compile(optimizer=Adam(lr=init_lrs, beta_1=0.8), loss=loss, metrics=metrics)
    model.summary()
    fit()

    filepath_swa = model_dir + filepath.split("/")[-1].split(".")[0] + "-swa.h5"
    model.save_weights(filepath_swa)

    # 保存配置，方便多模型集成
    save_config(filepath, cfg)
    #save_config(filepath_swa, cfg)

cfgs = ("dssm", "word", 20,'gensim', 256, [], 20, earlystop_patience) # 55s

def train_all_models():
    cfg = cfgs
    K.clear_session()
    model = get_model(cfg, None)
    swa_model = get_model(cfg, None)
    train_model(model, swa_model, cfg)


#####################################################################
#                         模型评估、模型融合、模型测试
#####################################################################

evaluate_path = model_dir + "y_pred.pkl"


def evaluate_models():
    train_y_preds, test_y_preds = [], []
    all_cfgs = json.loads(open(configs_path, 'r', encoding="utf8").read())
    print('---all_cfgs:',all_cfgs)
    num_clfs = len(all_cfgs)
    print('num_clfs',num_clfs)
    for weight, cfg in all_cfgs.items():
        K.clear_session()
        model_type, dtype, input_length, ebed_type, w2v_length, n_hidden, n_epoch, patience = cfg
        data = load_data(dtype, input_length, w2v_length)
        train_x, train_y, test_x, test_y = split_data(data)
        model = get_model(cfg, weight)
        train_y_preds.append(model.predict(train_x, batch_size=test_batch_size).reshape(-1))
        test_y_preds.append(model.predict(test_x, batch_size=test_batch_size).reshape(-1))
        r_train, f1_train, train_thresh = r_f1_thresh(train_y_preds, train_y)
        r_test, f1_test, test_thresh = r_f1_thresh(test_y_preds, test_y)
    print('f1_train:',f1_train)
    print('f1_test',f1_test)
    #train_y_preds, test_y_preds = np.array(train_y_preds), np.array(test_y_preds)
    #pd.to_pickle([train_y_preds, train_y, test_y_preds, test_y], evaluate_path)


blending_path = model_dir + "blending_gdbm.pkl"


def train_blending():
    """ 根据配置文件和验证集的值计算融合模型 """
    train_y_preds, train_y, valid_y_preds, valid_y = pd.read_pickle(evaluate_path)
    train_y_preds = train_y_preds.T
    valid_y_preds = valid_y_preds.T

    '''融合使用的模型'''
    #clf = LogisticRegression()
    #clf.fit(valid_y_preds, valid_y)

    #train_y_preds_blend = clf.predict_proba(train_y_preds)[:, 1]
    r, f1_train, train_thresh = r_f1_thresh(train_y_preds, train_y)

    #valid_y_preds_blend = clf.predict_proba(valid_y_preds)[:, 1]
    r, f1_test, valid_thresh = r_f1_thresh(valid_y_preds, valid_y)
    print(f1_train,f1_test)
    #pd.to_pickle(((train_thresh + valid_thresh) / 2, clf), blending_path)


def result():
    global df1
    all_cfgs = json.loads(open(configs_path, 'r', encoding="utf8").read())
    num_clfs = len(all_cfgs)
    test_y_preds = []
    X = {}
    for cfg in all_cfgs.values():
        model_type, dtype, input_length, ebed_type, w2v_length, n_hidden, n_epoch, patience = cfg
        key_ = f"{dtype}_{input_length}"
        if key_ not in X: X[key_] = input_data(df1["sent1"], df1["sent2"], dtype=dtype, input_length=input_length)

    for weight, cfg in all_cfgs.items():
        K.clear_session()
        model_type, dtype, input_length, ebed_type, w2v_length, n_hidden, n_epoch, patience = cfg
        key_ = f"{dtype}_{input_length}"
        model = get_model(cfg, weight)
        test_y_preds.append(model.predict(X[key_], batch_size=test_batch_size).reshape(-1))

    test_y_preds = np.array(test_y_preds).T
    thresh, clf = pd.read_pickle(blending_path)
    result = clf.predict_proba(test_y_preds)[:, 1].reshape(-1) > thresh

    df_output = pd.concat([df1["id"], pd.Series(result, name="label", dtype=np.int32)], axis=1)

    topai(1, df_output)


# 文档第二步，训练多个不同的模型，index取值为0-6

train_all_models()

# 文档第三步，训练blending模型
# if False:
evaluate_models()
#train_blending()

# 文档第四步，测试blending模型
# if False:
# result()
#Total params: 201,228,449