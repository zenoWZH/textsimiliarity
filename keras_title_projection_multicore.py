# coding: utf-8
from __future__ import print_function

"""
created on 2017/07/06
@author: liuenda
"""

import gc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.utils import multi_gpu_model

from sklearn.metrics import classification_report
from gensim import corpora, models, similarities
from sklearn import datasets
from sklearn import linear_model
from sklearn import svm
from gensim.models import word2vec
from sklearn import preprocessing
import random
import pickle
import time
from keras.preprocessing import sequence
from keras.layers import Input, Embedding, LSTM, Dense, Dropout, Bidirectional
from keras.models import Model

# from copy import deepcopy
base_path = "/home/R2016hwang/research/"
model_name_en = base_path + "word2vec/model_CBOW_en_200_wzh.w2v"
model_name_jp = base_path + "word2vec/model_CBOW_jp_200_wzh.w2v"
model_name_zh = base_path + "word2vec/model_CBOW_zh_200_wzh.w2v"

model_en = word2vec.Word2Vec.load(model_name_en)  # type: object
model_jp = word2vec.Word2Vec.load(model_name_jp)
model_zh = word2vec.Word2Vec.load(model_name_zh)

trans_jp_en = np.load("word2vec/jp_en_200.npy")
trans_zh_en = np.load("word2vec/zh_en_200.npy")

# datapath = base_path + "data_prepare/cleaned_jp_en_zh.csv"
# base_path = "/home/M2015eliu/cas/2017.1.1~LiuSTM/"
# model_name_en = base_path + "data/model-en/W2Vmodle.bin"
# model_name_jp = base_path + "data/model-jp/W2Vmodle.bin"

model_en = word2vec.Word2Vec.load(model_name_en)
model_jp = word2vec.Word2Vec.load(model_name_jp)

maxlen = 30  # Default: 0 -> infinite
epoch = 10
dim_lstm = 200
dim_1 = 800
# dim_2 = 100
# dim_3 = 50
dropout_rate = 0.0
bias_y = 0
loss_function = "mse"
mode = "reg"  # reg, binary
rnn_type = "bi-lstm"  # lstm, bi-lstm
bi_lstm_mode = "sum"  # concat, sum
print("maxlen", maxlen, "epoch", epoch, "dim_lstm", dim_lstm)
print("dim_Dense", dim_1)
print("dropout_rate", dropout_rate, ", LSTM type:", rnn_type, bi_lstm_mode)
p_activation = ["relu", "relu", "relu"]
print("Activation function:", p_activation)
print("bias of y:", bias_y)
print("loss_function:", loss_function)
start = 0
step = 10
print("start:", start, "end:", )
print("------------------------------")
random.seed(1234)

" Padding the sequence"
INPUT_SIZE = dim_lstm

def padding(sequence, maxlen=maxlen, padding_value=0.0):
    np_sequance = np.array(sequence)
    #print(np_sequance.shape)
    if np_sequance.shape[0] == 0:
        #return float('nan')
        print("长度为零")
        #return float('nan')
        return np.zeros((maxlen, INPUT_SIZE))
    if np_sequance.shape[0] < maxlen:
        z = np.zeros((maxlen, INPUT_SIZE))
    #    print(z.shape)
        z[:np_sequance.shape[0], :np_sequance.shape[1]] = np_sequance
    else:
        z = np_sequance[:maxlen, :]
    return z


"""
Find the ranking results with respect to real pairs
Defaulty, projection1 should be JP
Whiile, projection2 should be EN->JP
"""


def find_ranking_batch(projection1, projection2, dlmodel, batch=10):
    sim_results = []
    rank_results = []
    sample_length = len(projection2)
    batch = len(projection2)
    start_time = time.time()
    print("Find ranking of ",sample_length," sentence pairs")

    # Iterate each of the ariticle from projection1 (999) as proj1
    # Calculate the simialrity of proj1 with all ariticles in projection2 (999)
    # for i, proj1 in enumerate(projection1):
    for i in range(0, sample_length, batch):
        print("Find answer for doc.", i, i + batch)
        proj1 = projection1[i:i + batch]

        proj1_tile = np.repeat(proj1, sample_length, axis=0)
        proj2_tile = np.tile(projection2, (batch, 1, 1))
        #print(proj1_tile.shape)
        #print(proj2_tile.shape)

        # For each batch, we should tile each of the element
        sim = dlmodel.predict([proj1_tile, proj2_tile])[:, 0]
        for j in range(0, batch):
            rank = pd.Series(sim[j*sample_length:(j+1)*sample_length]).rank(ascending=False)[i+j]
            sim_results.append(sim)
            rank_results.append(rank)

        del proj1,proj1_tile,proj2_tile
        gc.collect()
    # sim_results contains 999*999 similairty matrix
    print("Use time(s)",time.time()-start_time)
    return sim_results, rank_results


"""
Find the ranking results with respect to real pairs
Defaulty, projection1 should be JP
Whiile, projection2 should be EN->JP
"""


def find_ranking_quick(projection1, projection2, dlmodel):
    sim_results = []
    rank_results = []

    # ---- Prepare the model_1 ---- #

    # Input layer
    input_1 = Input(shape=(maxlen, 200), dtype='float32', name='main_input_1')
    input_2 = Input(shape=(maxlen, 200), dtype='float32', name='main_input_2')

    # LSTM layer
    if rnn_type == "lstm":
        lstm_out_1 = LSTM(dim_lstm, go_backwards=True)(input_1)
        lstm_out_2 = LSTM(dim_lstm, go_backwards=True)(input_2)
    elif rnn_type == "bi-lstm":
        lstm_out_1 = Bidirectional(LSTM(dim_lstm, go_backwards=True), merge_mode=bi_lstm_mode)(input_1)
        lstm_out_2 = Bidirectional(LSTM(dim_lstm, go_backwards=True), merge_mode=bi_lstm_mode)(input_2)

    # Model definition
    model_1 = multi_gpu_model(input=[input_1, input_2], output=[lstm_out_1, lstm_out_2])

    # Compile the model
    if mode == "reg":
        model_1.compile(optimizer='adam', loss=loss_function)
    else:
        model_1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Show the structure of the model
    # print model_1.summary()

    # Set the weights
    model_1.layers[2].set_weights(dlmodel.layers[2].get_weights())
    model_1.layers[3].set_weights(dlmodel.layers[3].get_weights())

    # Find the real projection
    v1, v2 = model_1.predict([projection1, projection2])

    # ---- Prepare the model_2 ----

    # Input layer
    if rnn_type == "lstm":
        co = 1
    elif rnn_type == "bi-lstm":
        co = 2
    if bi_lstm_mode == "sum" or bi_lstm_mode == "mul":
        co = 1
    input_1 = Input(shape=(dim_lstm * co,), dtype='float32', name='lstm1')
    input_2 = Input(shape=(dim_lstm * co,), dtype='float32', name='lsmt2')

    # Merge layer
    merged_vector = keras.layers.concatenate([input_1, input_2], axis=-1)

    # (Dense 1) * 3
    x1 = Dense(dim_1, activation=p_activation[0])(merged_vector)
    x1 = Dropout(dropout_rate)(x1)

    # x1 = Dense(dim_2, activation=p_activation[1])(x1)
    # x1 = Dropout(dropout_rate)(x1)
    #
    # x1 = Dense(dim_3, activation=p_activation[2])(x1)
    # x1 = Dropout(dropout_rate)(x1)

    # main_output = Dense(1, activation='sigmoid', name='main_output')(x1)
    if mode == "reg":
        main_output = Dense(1, name='main_output')(x1)
        # main_output = Dense(1, activation="sigmoid", name='main_output')(x1)
    else:
        main_output = Dense(2, activation='softmax', name='main_output')(x1)

    # Model definition
    model_2 = multi_gpu_model(input=[input_1, input_2], output=main_output)

    # Compile the model
    if mode == "reg":
        model_2.compile(optimizer='adam', loss=loss_function)
    else:
        model_2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Show the structure of the model
    # print model_2.summary()

    # Find the real projection
    v3 = model_2.predict([v1, v2])

    # Set the weights
    model_2.layers[3].set_weights(dlmodel.layers[5].get_weights())
    model_2.layers[5].set_weights(dlmodel.layers[7].get_weights())
    # model_2.layers[7].set_weights(dlmodel.layers[9].get_weights())
    #
    # model_2.layers[9].set_weights(dlmodel.layers[11].get_weights())

    projection1 = v1
    projection2 = v2

    # Iterate each of the ariticle from projection1 (999) as proj1
    # Calculate the simialrity of proj1 with all ariticles in projection2 (999)
    for i, proj1 in enumerate(projection1):
        # print("Find answer for doc.", i)
        proj1_tile = np.tile(proj1, (len(projection2), 1))
        sim = model_2.predict([proj1_tile, projection2])[:, 0]
        rank = pd.Series(sim).rank(ascending=False)[i]
        sim_results.append(sim)
        rank_results.append(rank)

    # release memory
    del input_1, input_2, model_1, model_2, lstm_out_1, lstm_out_2, main_output, projection1, projection2, v1, v2, v3
    gc.collect()

    # sim_results contains 999*999 similairty matrix
    return sim_results, rank_results


"""
Find the ranking results with respect to real pairs
Defaulty, projection1 should be JP
Whiile, projection2 should be EN->JP
"""


def find_ranking(projection1, projection2, dlmodel):
    sim_results = []
    rank_results = []

    # Iterate each of the ariticle from projection1 (999) as proj1
    # Calculate the simialrity of proj1 with all ariticles in projection2 (999)
    for i, proj1 in enumerate(projection1):
        print("Find answer for doc.", i)
        proj1_tile = np.tile(proj1, (len(projection2), 1, 1))
        sim = dlmodel.predict([proj1_tile, projection2])[:, 0]
        rank = pd.Series(sim).rank(ascending=False)[i]
        sim_results.append(sim)
        rank_results.append(rank)

    # sim_results contains 999*999 similairty matrix
    return sim_results, rank_results


"""
rank_results should be list of (999,)
"""


def find_top(rank_results, top):
    s = pd.Series(rank_results)
    n_top = (s <= top).sum()
    return n_top


def average_docment(document_embedding):
    return np.average(document_embedding, axis=0)


def sum_docment(document_embedding):
    return np.sum(document_embedding, axis=0)


# def doc2feature(corpus, tfidf, dictionary, w2v):
#     doc_features = []
#     for index, doc_bof in enumerate(corpus):
#
#         if index % 1000 == 0:
#             print(index)
#
#         doc_tfidf = tfidf[doc_bof]
#
#         doc_feature = np.zeros((200,))
#
#         for (token_id, token_tfidf) in doc_tfidf:
#             token = dictionary.get(token_id, "[unknown-id]").encode("utf-8")
#             # if token in w2v:
#             if True:
#                 token_w2v = w2v[token]
#             else:
#                 print("No word:", token)
#                 continue
#             doc_feature += token_w2v * token_tfidf
#
#         average = True
#         if average:
#             doc_feature = np.true_divide(doc_feature, len(doc_tfidf))
#         doc_features.append(doc_feature)
#
#     return doc_features


def doc2vec_en(doc):
    # r = [model_en[token] for token in doc.split()]
    r = []
    r_failed = []

    for token in doc.split():
        if token in model_en:
            r.append(model_en[token])
        else:
            r_failed.append(token)

    # if len(r_failed) != 0:
    #     print " ".join(r_failed)

    return r


def doc2vec_jp(doc):
    # r = [model_en[token] for token in doc.split()]
    r = []
    r_failed = []

    for token in doc.split():
        if token in model_jp:
            r.append(model_jp[token])
        else:
            r_failed.append(token)

    # if len(r_failed) != 0:
    #     print " ".join(r_failed)

    return r


def doc2embed(doc, model, translation_matrix=None):
    # r = [model_en[token] for token in doc.split()]
    r = []
    r_failed = []

    for token in str(doc).split(' '):
        if token in model:
            if type(translation_matrix) != type(None):
                # print("translation_matrix ",translation_matrix.shape)
                # print(np.array(model[token]).shape)
                r.append(np.array(model[token]).dot(translation_matrix))
            else:
                r.append(model[token])
        else:
            r_failed.append(token)

    # if len(r_failed) != 0:
    #     print " ".join(r_failed)
    # print("Failed Number",len(r_failed))
    return r


# def prepare_train(dir_en, dir_jp):
def prepare_train(dir_en_jp, second_language="jp", start=None, end=None):
    # df_en_mapping = pd.read_csv(dir_en)
    # df_jp_mapping = pd.read_csv(dir_jp)
    second_article = second_language + "_article"

    df_en_jp = pd.read_csv(dir_en_jp,
                           names=["HEADLINE_ALERT_TEXT_x", "HEADLINE_ALERT_TEXT_y", "HEADLINE_ALERT_TEXT", "jp_article",
                                  "en_article", "zh_article"],
                           header=0)
    df_en_jp = df_en_jp.dropna(axis=0, how='any')
    df_en_mapping = df_en_jp[["en_article"]].iloc[start:end]
    df_jp_mapping = df_en_jp[[second_article]].iloc[start:end]

    print("Reading English Data:", len(df_en_mapping))
    print("Reading " + second_language + " Data:", len(df_jp_mapping))

    sample_size = len(df_en_mapping)

    assert len(df_en_mapping) == len(df_jp_mapping)

    # Convert mapping to list type and then concat to the a list
    print("Merging the English and Japanes news dataframe...")
    df_train_1 = pd.concat([df_en_mapping, df_jp_mapping], axis=1)
    df_train_1['similarity'] = pd.Series(np.ones(sample_size, ) * 5)
    df_train_1['dis_similarity'] = pd.Series(np.ones(sample_size, ) * 1)

    # Remove null line
    print("Drop the null line...")
    # df_train_1 = df_train_1.dropna(subset=['en_article'])
    df_train_1 = df_train_1[df_train_1['en_article'] != '<NULL>']

    # Expand the training data
    en_article_wrong = df_train_1.en_article.iloc[random.sample(range(len(df_train_1)), len(df_train_1))]
    en_article_wrong.index = df_train_1.index
    print((en_article_wrong == df_train_1.en_article).value_counts())
    df_train_1['en_article_wrong'] = en_article_wrong

    # Convert dateframe to list
    train_1 = df_train_1[['en_article', second_article, 'similarity']].values.tolist()
    train_2 = df_train_1[['en_article_wrong', second_article, 'dis_similarity']].values.tolist()

    return train_1, train_2, df_train_1


if __name__ == "__main__":

    input = 2
    # k = 10

    # --- Prepare and Loading the training data --- #

    # if input == 1:
    #     # Prepare For the training data
    #     sample_size = "_1000"
    #     dir_en = "./data/mapping/en_mapped_"+str(k) + sample_size + ".csv"
    #     dir_jp = "./data/mapping/jp_mapped_" + str(k) + sample_size + ".csv"
    #
    #     # Prepare For the test data
    #     sample_size = "_1k2k"
    #     dir_en_test = "./data/mapping/en_mapped_"+str(k) + sample_size + ".csv"
    #     dir_jp_test = "./data/mapping/jp_mapped_" + str(k) + sample_size + ".csv"
    #
    #     train_1, train_2, df_train_1 =     prepare_train(dir_en, dir_jp)
    #     test_1, test_2, df_test_1 = prepare_train(dir_en_test, dir_jp_test)

    if input == 2:
        # split_line = 5000
        # end_line = 6000
        # Prepare For the training data
        # dir_en = base_path + "en_news.csv"
        # dir_jp = base_path + "jp_news.csv"
        dir_en_jp = base_path + "data_prepare/cleaned_jp_en_zh.csv"

        # pairs_correct, pairs_wrong, df_pairs = prepare_train(dir_en, dir_jp)
        # pairs_correct, pairs_wrong, df_pairs = prepare_train(dir_en_jp)
        _1, _2, df_pairs_enjp = prepare_train(dir_en_jp)
        # 这里的 pairs_wrong 和 pairs_correct 都 不使用！
        # train_1 = pairs_correct[0:2000] + pairs_correct[3000:5000]
        # test_1 = pairs_correct[2000:3000]

        # train_2 = pairs_wrong[0:2000] + pairs_wrong[3000:5000]
    # test_2 = pairs_wrong[split_line:end_line]

    # Expand the training data
    # train = train_1 + train_2

    # ------- Independent data sets -------
    # print("Using the new test data to evaluate.......")
    # df_pairs_evaluate = df_pairs.iloc[50000:55000:5]

    # df_pairs_evaluate['word2vec_en'] = df_pairs_evaluate['en_article'].apply(doc2vec_en)
    # df_pairs_evaluate['word2vec_jp'] = df_pairs_evaluate['jp_article'].apply(doc2vec_jp)

    # df_pairs_evaluate['padding_en'] = df_pairs_evaluate['word2vec_en'].apply(padding)
    # df_pairs_evaluate['padding_jp'] = df_pairs_evaluate['word2vec_jp'].apply(padding)

    # features_en_new = np.stack(df_pairs_evaluate["padding_en"].values)
    # features_jp_new = np.stack(df_pairs_evaluate["padding_jp"].values)

    # --- Apply the word2vec model to the data sets --- #

    # df_pairs_sample = df_pairs.iloc[0:5000]

    # df_pairs_sample['word2vec_en'] = df_pairs_sample['en_article'].apply(doc2vec_en)
    # df_pairs_sample['word2vec_jp'] = df_pairs_sample['jp_article'].apply(doc2vec_jp)

    # ---- Padding the vector ---- #
    # df_pairs_sample['padding_en'] = df_pairs_sample['word2vec_en'].apply(padding)
    # df_pairs_sample['padding_jp'] = df_pairs_sample['word2vec_jp'].apply(padding)

    # --- Prepare the training data --- #

    # Generate training data (similarity = 1)
    # features_en_1 = np.stack(df_pairs_sample["padding_en"].values)
    # features_jp_1 = np.stack(df_pairs_sample["padding_jp"].values)

    # Generate training data (similarity = 0)
    # features_en_0 = np.array(features_en_1)
    # np.random.shuffle((features_en_0))

    ##############################new#########################################
    print("Using the new test data to evaluate.......")
    TIME_STEP = maxlen
    df_pairs_evaluate = df_pairs_enjp.iloc[50000:55000:5]

    df_pairs_evaluate['word2vec_en'] = df_pairs_evaluate['en_article'].apply(doc2embed, args=(model_en,))
    #df_pairs_evaluate['word2vec_en_wrong'] = df_pairs_evaluate['en_article_wrong'].apply(doc2embed, args=(model_en,))
    df_pairs_evaluate['word2vec_jp'] = df_pairs_evaluate['jp_article'].apply(doc2embed, args=(model_jp, trans_jp_en))

    df_pairs_evaluate['padding_en'] = df_pairs_evaluate['word2vec_en'].apply(padding, args=(TIME_STEP,))
    #df_pairs_evaluate['padding_en_wrong'] = df_pairs_evaluate['word2vec_en_wrong'].apply(padding, args=(TIME_STEP,))
    df_pairs_evaluate['padding_jp'] = df_pairs_evaluate['word2vec_jp'].apply(padding, args=(TIME_STEP,))

    df_pairs_evaluate.dropna(axis=0, how='all')

    features_en_new = np.stack(df_pairs_evaluate["padding_en"].values)
    features_jp_new = np.stack(df_pairs_evaluate["padding_jp"].values)
    #features_en_new_wrong = np.stack(df_pairs_evaluate["padding_en_wrong"].values)

    # ---- Training data ---- #
    df_pairs_sample = df_pairs_enjp.iloc[0:50000]

    df_pairs_sample['word2vec_en'] = df_pairs_sample['en_article'].apply(doc2embed, args=(model_en,))
    #df_pairs_sample['word2vec_en_wrong'] = df_pairs_sample['en_article_wrong'].apply(doc2embed, args=(model_en,))
    df_pairs_sample['word2vec_jp'] = df_pairs_sample['jp_article'].apply(doc2embed, args=(model_jp, trans_jp_en))

    # ---- Padding the vector ---- #
    df_pairs_sample['padding_en'] = df_pairs_sample['word2vec_en'].apply(padding, args=(TIME_STEP,))
    #df_pairs_sample['padding_en_wrong'] = df_pairs_sample['word2vec_en_wrong'].apply(padding, args=(TIME_STEP,))
    df_pairs_sample['padding_jp'] = df_pairs_sample['word2vec_jp'].apply(padding, args=(TIME_STEP,))

    df_pairs_sample.dropna(axis=0, how='all')

    df_pairs_sample = df_pairs_sample[df_pairs_sample['word2vec_en'] != float('nan')]

    # Generate training data (similarity = 1)
    features_en_1 = np.array(df_pairs_sample["padding_en"].values.tolist(), dtype=np.float32)
    features_jp_1 = np.array(df_pairs_sample["padding_jp"].values.tolist(), dtype=np.float32)

    # Generate training data (similarity = 0)
    #features_en_0 = np.array(df_pairs_sample["padding_en_wrong"].values.tolist(), dtype=np.float32)
    features_en_0 = np.array(features_en_1)
    np.random.shuffle((features_en_0))

    #####################################new############################################
    
    # check the duplicated amount
    c = np.all(features_en_1 == features_en_0, axis=(1, 2))
    print("C value =", c.sum(), "position:", np.where(c == True)[0].tolist())
    
    # ------------------- Model ------------------ #

    # Input layer
    input_1 = Input(shape=(maxlen, 200), dtype='float32', name='main_input_1')
    input_2 = Input(shape=(maxlen, 200), dtype='float32', name='main_input_2')

    # LSTM layer
    # lstm_out_1 = LSTM(50)(input_1)
    # lstm_out_2 = LSTM(50)(input_2)
    if rnn_type == "lstm":
        lstm_out_1 = LSTM(dim_lstm, go_backwards=True)(input_1)
        lstm_out_2 = LSTM(dim_lstm, go_backwards=True)(input_2)
    elif rnn_type == "bi-lstm":
        lstm_out_1 = Bidirectional(LSTM(dim_lstm, go_backwards=True), merge_mode=bi_lstm_mode)(input_1)
        lstm_out_2 = Bidirectional(LSTM(dim_lstm, go_backwards=True), merge_mode=bi_lstm_mode)(input_2)

    # Merge layer
    merged_vector = keras.layers.concatenate([lstm_out_1, lstm_out_2], axis=-1)

    # (Dense 1) * 3
    x1 = Dense(dim_1, activation=p_activation[0])(merged_vector)
    x1 = Dropout(dropout_rate)(x1)

    # x1 = Dense(dim_2, activation=p_activation[1])(x1)
    # x1 = Dropout(dropout_rate)(x1)
    #
    # x1 = Dense(dim_3, activation=p_activation[2])(x1)
    # x1 = Dropout(dropout_rate)(x1)

    if mode == "reg":
        main_output = Dense(1, name='main_output')(x1)
    # main_output = Dense(1, activation="sigmoid", name='main_output')(x1)
    else:
        main_output = Dense(2, activation='softmax', name='main_output')(x1)

    # Model definition
    model_lstm2 = multi_gpu_model(input=[input_1, input_2], output=main_output)

    # Compile the model
    # model_lstm2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model_lstm2.compile(optimizer='adam', loss=loss_function)
    # model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Show the structure of the model
    print(model_lstm2.summary())
    
    # ------------------- Model End ------------------ #
    
    # ------------------- Train ------------------ #
    
    # divide data into 10 parts, in order to evaluate over fitting
    train_loss = []
    test_loss = []
        #saved_model = []
    tops = {}
    tops_test = {}
    # Initialize the top saver
    for i in [1, 5, 10]:
        tops[i] = []
        tops_test[i] = []

    div = 10
    epoch = 10
    for ep in range(epoch):
        print("Epoch: ", ep+1, "/", epoch, "----------------------------------------\n")
        for sp in range(div):
            print("Data parts ",sp+1,"/",div,"----------------------------------------\n")
            rag_left = 5000*sp
            rag_right = 5000*(sp+1)
            # Prepare the final training and test data
            X_1 = np.concatenate((features_en_1[rag_left:rag_right], features_en_0[rag_left:rag_right]), axis=0)
            X_2 = np.concatenate((features_jp_1[rag_left:rag_right], features_jp_1[rag_left:rag_right]), axis=0)
            if mode == "reg":
                y = np.concatenate((np.ones(len(features_en_1[rag_left:rag_right])), np.zeros(len(features_en_0[rag_left:rag_right])) + bias_y), axis=0)
            # y = np.concatenate((np.ones(len(features_en_1)), np.zeros(len(features_en_0))), axis = 0)
            else:
                y = np.concatenate((np.ones(len(features_en_1[rag_left:rag_right])), np.zeros(len(features_en_0[rag_left:rag_right]))), axis=0)

            # --- Split into test data and training data --- #

            X1_train1, X1_test_1, X1_train2, X1_train3_wrong, X1_test_0 = np.split(X_1, [2500, 3000, 5000, 9500])
            X2_train1, X2_test_1, X2_train2, X2_train3_wrong, X2_test_0 = np.split(X_2, [2500, 3000, 5000, 9500])
            y_train1, y_test, y_train2, y_train3_wrong, Y_o = np.split(y, [2500, 3000, 5000, 9500])

            #X1_train1, X1_test_1, X1_train2, X1_train3_wrong, X1_test_0 = np.split(X_1, [2000, 3000, 5000, 9000])
            #X2_train1, X2_test_1, X2_train2, X2_train3_wrong, X2_test_0 = np.split(X_2, [2000, 3000, 5000, 9000])
            #y_train1, y_test, y_train2, y_train3_wrong, Y_o = np.split(y, [2000, 3000, 5000, 9000])

            X1_train = np.concatenate((X1_train1, X1_train2, X1_train3_wrong), axis=0)
            X2_train = np.concatenate((X2_train1, X2_train2, X2_train3_wrong), axis=0)
            y_train = np.concatenate((y_train1, y_train2, y_train3_wrong), axis=0)
            # X_train_correct = np.concatenate((X_train1, X_train2), axis = 0)
            # y_train_correct = np.concatenate((y_train1, y_train2), axis = 0)

            # --- Generate balanced test data --- #
            X1_test = np.concatenate((X1_test_1, X1_test_0), axis=0)
            X2_test = np.concatenate((X2_test_1, X2_test_0), axis=0)
            if mode == "reg":
                y_test = np.concatenate((np.ones(len(X1_test_1)), np.zeros(len(X1_test_0)) + bias_y), axis=0)
            # y_test = np.concatenate((np.ones(len(X1_test_1)), np.zeros(len(X1_test_0))), axis = 0)
            else:
                y_test = np.concatenate((np.ones(len(X1_test_1)), np.zeros(len(X1_test_0))), axis=0)

            # --- Generate balanced test data 2 --- #

            # Generate training data (similarity = 0)
            features_en_0_new = np.array(features_en_new)
            np.random.shuffle((features_en_0_new))

            # check the duplicated amount
            c = np.all(features_en_new == features_en_0_new, axis=(1, 2))
            print("C value =", c.sum(), "position:", np.where(c == True)[0].tolist())

            # Prepare the final training and test data
            X1_test_new = np.concatenate((features_en_new, features_en_0_new), axis=0)
            X2_test_new = np.concatenate((features_jp_new, features_jp_new), axis=0)

            if mode == "reg":
                y_test_new = np.concatenate((np.ones(len(features_en_new)), np.zeros(len(features_en_0_new)) + bias_y), axis=0)
            # y_test = np.concatenate((np.ones(len(X1_test_1)), np.zeros(len(X1_test_0))), axis = 0)
            else:
                y_test_new = np.concatenate((np.ones(len(features_en_new)), np.zeros(len(features_en_0_new))), axis=0)




            # Fit the training model

            hist = model_lstm2.fit([X1_train, X2_train], [y_train],
                               validation_data=([X1_test, X2_test], y_test),
                               epochs=ep+1,
                               batch_size=100,
                               initial_epoch=ep)
            
            #print(hist.history)
            train_loss.append(hist.history['loss'])
            test_loss.append(hist.history['val_loss'])

            # Evaluation on test data 1
            sim_results_test, rank_results_test = find_ranking_quick(X1_test,
                                                                     X2_test,
                                                                     model_lstm2)

            # Evaluation on test data 2
            sim_results_test2, rank_results_test2 = find_ranking_quick(features_en_new,
                                                                 features_jp_new,
                                                                 model_lstm2)
            
            
            # Save the mdoel
            #saved_model.append(model_lstm2.get_weights())

            # Evaluation

            for i in [1,5,10]:
                top = (pd.Series(rank_results_test) <= i).sum()
                tops[i].append(top)
                print("TOP", i, top)

                top_test = (pd.Series(rank_results_test2) <= i).sum()
                tops_test[i].append(top_test)
                print("[T] TOP", i, top_test)

            # release memory
            del rank_results_test, rank_results_test2, sim_results_test, sim_results_test2
            gc.collect()
    # # Save the history and the model
    #code = "proj"
    #path_model_lstm2 = "model_lstm2_" + code
    #model_lstm2.save(path_model_lstm2)
    #path_hist = "hist_lstm2_" + code
    #f = open(path_hist, "wb")
    #pickle.dump(hist.history, f)
    #f.close()

            # ---- 看样子主要的时间都花在了lstm上吧？ --- #
            # 想办法保存lstm以后的映射 --- #
            
    plt.plot(train_loss)
    plt.plot(test_loss)
    
        # --- find ranking --- #
        # sim_results_test_slow, rank_results_test_slow = find_ranking(X1_test_1, X2_test_1, model_lstm2)
        # sim_results_test, rank_results_test = find_ranking_quick(X1_test_1, X2_test_1, model_lstm2)
        # print("TOP1", (pd.Series(rank_results_test) <= 1).sum())
        # print("TOP5", (pd.Series(rank_results_test) <= 5).sum())
        # print("TOP10", (pd.Series(rank_results_test) <= 10).sum())

        # # --- Prepare for a new independent evaluation balanced data --- #
        #
        # df_pairs_evaluate = df_pairs.iloc[55000:60000]
        #
        # df_pairs_evaluate['word2vec_en'] = df_pairs_evaluate['en_article'].apply(doc2vec_en)
        # df_pairs_evaluate['word2vec_jp'] = df_pairs_evaluate['jp_article'].apply(doc2vec_jp)
        #
        # features_en_eva = doc2feature(corpus_en[60000:61000], tfidf_en, dictionary_en, model_en)
        # features_jp_eva = doc2feature(corpus_jp[60000:61000], tfidf_jp, dictionary_jp, model_jp)
        #
        # features_merge_eva = np.concatenate((features_en_eva,features_jp_eva), axis = 1)
        #
        # features_en_wrong_eva =  features_en[:1000]
        # # features_en_wrong_eva = np.array(features_en_eva)
        # # np.random.shuffle((features_en_wrong_eva))
        # # c = np.all(features_en_wrong_eva == features_en_eva, axis=1)
        # # print "C value =", c.sum() # check the duplicated amount
        #
        # features_merge_wrong = np.concatenate((features_en_wrong_eva,features_jp_eva), axis = 1)
        #
        # X_eva = np.concatenate((features_merge_eva, features_merge_wrong), axis = 0)
        # y_eva = np.concatenate((np.ones(len(features_merge_eva)), np.zeros(len(features_en_wrong_eva))), axis = 0)
        #
        # y_eva_predict = clf.predict(X_eva)
        #
        # print("classification report of TRAINING data:")
        # print(classification_report(y_eva, y_eva_predict))
        #
        #
        # # --- Evaluation for SVM --- #
        #
        # y_test_proba = clf.predict_proba(X_test)
        # y_train_proba = clf.predict_proba(X_train)
        #
        # # sim_results_train, rank_results_train = find_ranking(projection1_train, projection2_train)
        # sim_results_test, rank_results_test = find_ranking(X_test[:,:200] ,X_test[:,200:], clf)
        #
        #
        # print(pd.Series(rank_results_test).describe())
