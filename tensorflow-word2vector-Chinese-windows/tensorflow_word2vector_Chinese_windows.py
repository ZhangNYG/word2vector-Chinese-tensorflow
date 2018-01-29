# -*-coding:utf-8 -*-

# ***************Word2vec For Chinese**********************#
# Revised on January 25, 2018 by
# Author: XianjieZhang
# Dalian University of Technology
# email: xj_zh@foxmail.com
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import zipfile

import numpy as np
import time
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
#中文图片乱码解决
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc")

filename = "135MB.txt"


def read_data(filename):
    with open(filename, encoding='utf-8') as f:
        data = []
        counter = 0
        for line in f:
            line = line.strip('\n').strip('')
            if line != "":
                counter += 1
                data_tmp = [word for word in line.split(" ") if word != '']
            data.extend(data_tmp)
            #print(data_tmp)
        print(counter)
    return data


words = read_data(filename)
print('Data size', len(words))

# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = 20000


# 建立数据字典
def build_dataset(words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


data, count, dictionary, reverse_dictionary = build_dataset(words)

#保存字典
f_dict = open('dictionary_data.txt', 'w', encoding='utf-8')
f_dict.write(str(reverse_dictionary))
f_dict.close()
#保存统计个数
f_count = open('count_data.txt', 'w', encoding='utf-8')
f_count.write(str(count))
f_count.close()

del words  # Hint to reduce memory.
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0


# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window  # 判断是否正常
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ] 3
    buffer = collections.deque(maxlen=span)  # span=3   deque是一个队列
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):  # 算术运算符" // "来表示整数除法，返回不大于结果的一个最大的整数
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels


# 全局变量data_index来记录当前取到哪了，每次取一个batch后会向后移动，如果超出结尾，则又从头开始。data_index = (data_index + 1) % len(data)
# skip_window是确定取一个词周边多远的词来训练，比如说skip_window是2，则取这个词的左右各两个词，来作为它的上下文词。后面正式使用的时候取值是1，也就是只看左右各一个词。
# 这里的num_skips我有点疑问，按下面注释是说，How many times to reuse an input to generate a label.，但是我觉得如果确定了skip_window之后，完全可以用
# 这边用了一个双向队列collections.deque，第一次遇见，看代码与list没啥区别，从网上简介来看，双向队列主要在左侧插入弹出的时候效率高，但这里并没有左侧的插入弹出呀，所以是不是应该老实用list会比较好呢？
# num_skips=2*skip_window来确定需要reuse的次数呀，难道还会浪费数据源不成？
batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]],
          '->', labels[i, 0], reverse_dictionary[labels[i, 0]])
# 上面代码只是给出一个例子
# 下面参数的给出才是正式模型的构建的开始
# Step 4: Build and train a skip-gram model.
# 下面还有一些细节要处理，以后优化参数
batch_size = 128  # 一次训练词的数量
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1  # How many words to consider left and right.
num_skips = 2  # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 8  # Number of negative examples to sample. num_sampled = 64  这个是负样本个数，正样本的个数在这是1(labels的第二个维度)，


def plot_with_labels(low_dim_embs, labels, filename):

    filename = filename + time.strftime("%Y-%m-%d--%H_%M_%S", time.localtime()) + '.png'
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom',
                     fontproperties=font)
    plt.savefig(filename)
    plt.cla()
    plt.close('all')




graph = tf.Graph()

with graph.as_default():
    # Input data.
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device('/cpu:0'):
        # Look up embeddings for inputs.
        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        # tf.nn.embedding_lookup函数的用法主要是选取一个张量里面索引对应的元素。
        # Construct the variables for the NCE loss
        nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights,
                       biases=nce_biases,
                       labels=train_labels,
                       inputs=embed,
                       num_sampled=num_sampled,
                       num_classes=vocabulary_size))


    #学习率按照指数减少
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(1.0,      #最初的学习率
                                               global_step , #现在学习步数
                                               50000,        #每50000步衰减一次
                                               0.99,         #衰减指数
                                               staircase=True)  #这个为TRUE的时候可以2000步调整一次


    # Construct the SGD optimizer using a learning rate of 1.0.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

    # 计算词之间的相似
    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(
        normalized_embeddings, valid_dataset)
    similarity = tf.matmul(
        valid_embeddings, normalized_embeddings, transpose_b=True)

    # Add variable initializer.
    init = tf.initialize_all_variables()

# Step 5: Begin training.
num_steps = 10000001

with tf.Session(graph=graph) as session:

    # # 恢复前一次训练

    saver = tf.train.Saver()
    ckpt_state = tf.train.get_checkpoint_state('save/')
    if ckpt_state != None:
        print("对上次训练进行恢复..........")
        print('上次训练模型路径:  ',ckpt_state.model_checkpoint_path)
        saver.restore(session,ckpt_state.model_checkpoint_path)
    else:
        # We must initialize all variables before we use them.
        init.run()
    print("Initialized")

    loss_all= []
    average_loss = 0
    for step in xrange(num_steps):
        batch_inputs, batch_labels = generate_batch(
            batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            loss_all.append(average_loss)
            print("Average loss at step ", step, ": ", average_loss)
            print("词语训练位置:", data_index)    #词语训练位置
            average_loss = 0
            learn_rate=session.run(learning_rate)
            print("当前学习率：",learn_rate)

        # Note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 10000 == 0:
            #每10000步保存一次模型
            if step != 0:
                save_path = saver.save(session,"save/SaveModel.ckpt",global_step = global_step)
                print('保存变量模型save/SaveModel:',tf.train.get_checkpoint_state('save/').model_checkpoint_path)
            #计算显示一次相似词汇
            sim = similarity.eval()
            for i in xrange(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = "Nearest to %s:" % valid_word
                for k in xrange(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = "%s   %s ," % (log_str, close_word)
                print(log_str)


        final_embeddings = normalized_embeddings.eval()
        #每1万步保存一次词向量 和 前500词汇图片
        if step % 10000 == 0:
            if step!=0:
                try:
                    np.save("vecorForTxt.npy", final_embeddings)
                    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
                    plot_only = 500
                    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
                    labels = [reverse_dictionary[i] for i in xrange(plot_only)]
                    plot_with_labels(low_dim_embs, labels, filename='tsne-135MB')
                except ImportError:
                    print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")


        #每100万步保存一次全部词汇图片
        if step % 1000000 == 0:
            if step != 0:
                try:
                    for i_word in xrange(1,vocabulary_size-1000,500):    #总字典词汇量减去1000
                        low_dim_embs = tsne.fit_transform(final_embeddings[i_word:i_word+500, :])
                        labels = [reverse_dictionary[i] for i in xrange(i_word, i_word + 500)]
                        plot_with_labels(low_dim_embs, labels, filename='tsne-135MB--words-star-:'+ str(i_word)+'--' )
                except ImportError:
                    print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")

print("Step5 over")
print(type(final_embeddings))


# numpy.save("filename.npy",a)
# 利用这种方法，保存文件的后缀名字一定会被置为.npy，这种格式最好只用
# numpy.load("filename")来读取。
# Step 6: Visualize the embeddings.
#字典也要保存
# 保存
#dict_name = {1: {1: 2, 3: 4}, 2: {3: 4, 4: 5}}
#f = open('temp.txt', 'w')
#f.write(str(dict_name))
#f.close()

# 读取
#f = open('temp.txt', 'r')
#a = f.read()
#dict_name = eval(a)
#f.close()