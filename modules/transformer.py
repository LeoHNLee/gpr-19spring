import tensorflow as tf
import numpy as np
import pandas as pd

def scaled_dot_product_attention(query, key, value, masked=False):
    # Attention(Q,K,V) = softmax(QKt / root dk)V
    # key vector의 dim
    key_dim_size = float(key.get_shape().as_list()[-1])
    # query와 key에 대해 attention map을 구성
    key = tf.transpose(key, perm=[0,2,1])
    outputs = tf.matmul(query, key)/tf.sqrt(key_dim_size)

    # masking
    if masked :
        # 마스킹할 영역을 준비하기 위해 초기화
        diag_vals = tf.ones_like(outputs[0,:,:])
        # 행렬을 하삼각행렬로 만들어 상삼각영역에 대해서는 패딩 처리
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
        # 배치 크기만큼 확장
        masks = tf.tile(tf.expand_dims(tril,0), [tf.shape(outputs)[0],1,1])

        # 패딩 영역에 할당할 아주 낮은 음의 값을 행렬로 만들어 둠
        paddings = tf.ones_like(masks)*(-2**32+1)
        outputs = tf.where(tf.equal(masks, 0), paddings, outputs)

    attention_map = tf.nn.softmax(outputs)
    # attention map을 가지고 value에 대한 가중합을 진행
    return tf.matmul(attention_map, value)

def multi_head_attention(query, key, value, num_units, heads, masked=False):
    # 입력한 Q, K, V에 대해 선형층을 거치도록 함
    query = tf.keras.layers.Dense(num_units, activation=tf.nn.relu)(query)
    key = tf.keras.layers.Dense(num_units, activation=tf.nn.relu)(key)
    value = tf.keras.layers.Dense(num_units, activation=tf.nn.relu)(value)

    # 선형층을 통과한 query, key, value에 대해 지정된 헤드 수만큼 입력하도록 피쳐들을 분리
    query = tf.concat(tf.split(query, heads, axis=-1), axis=0)
    key = tf.concat(tf.split(key, heads, axis=-1), axis=0)
    value = tf.concat(tf.split(value, heads, axis=-1), axis=0)

    # self-attention 연산
    attention_feature = scaled_dot_product_attention(query, key, value, masked)
    # 다시 나눠진 헤드에 대한 피처들을 하나로 다시 모음
    attn_outputs = tf.concat(tf.split(attention_feature, heads, axis=0), axis=-1)
    # linear layer를 거침
    attn_outputs = tf.keras.layers.Dense(num_units, activation=tf.nn.relu)(attn_outputs)
    return attn_outputs

def feed_forward(inputs, num_units):
    # 출력 디멘전은 입력 디멘젼과 동일해야 한다.
    feature_shape = inputs.get_shpae()[-1]
    # 활성화 함수는 첫 레이어에서만 적용한다.
    inner_layer = tf.keras.layers.Dense(num_units, activation=tf.nn.relu)(inputs)
    outputs = tf.keras.layers.Dense(feature_shape)(inner_layer)
    return outputs

def sublayer_connection(inputs, sublayer, dropout=0.2):
    # residual connection
    outputs = layer_norm(inputs+tf.keras.layers.Dropout(dropout)(sublayer))
    return outputs

def layer_norm(inputs, eps=1e-6):
    feature_shape = inputs.get_shape()[-1:]
    # 평균과 표준편차를 전달한다
    mean = tf.keras.backend.mean(inputs, [-1], keepdims=True)
    std = tf.keras.backend.std(inputs, [-1], keepdims = True)
    beta = tf.Variable(tf.zeros(feature_shape), trainable=False)
    gamma = tf.Variable(tf.ones(feature_shape), trainable=False)
    return gamma*(inputs-mean)/(std+eps)+beta

def encoder_module(inputs, model_dim, ffn_dim, heads):
    # self attention layer
    self_attn = sublayer_connection(inputs, multi_head_attention(inputs, inputs, inputs, model_dim, heads))
    # position-wise feedforward layer
    outputs = sublayer_connection(self_attn, feed_forward(self_attn,ffn_dim))
    return outputs

def encoder(inputs, model_dim, ffn_dim, heads, num_layers):
    outputs = inputs
    for i in range(num_layers):
        outputs = encoder_module(outputs, model_dim, ffn_dim, heads)
    return outputs

def decoder_module(inputs, encoder_outputs, model_dim, ffn_dim, heads):
    # mask self attention layer
    masked_self_attn = sublayer_connection(inputs, multi_head_attention(inputs, inputs, inputs, model_dim, heads, masked=True))
    # encoder-decoder embedding을 위한 self-attention layer
    self_attn = sublayer_connection(masked_self_attn, multi_head_attention(masked_self_attn, encoder_outputs, encoder_outputs, model_dim, heads))
    # position-wise feed-forward layer
    outputs = sublayer_connection(self_attn, feed_forward(self_attn, ffn_dim))
    return outputs

def positional_encoding(dim, sentence_length):
    encoded_vec = np.array([pos/np.power(10000, 2*i/dim) for pos in range(sentence_length) for i in range(dim)])
    encoded_vec[::2] = np.sin(encoded_vec[::2])
    encoded_vec[1::2] = np.cos(encoded_vec[1::2])
    return tf.constant(encoded_vec.reshape([sentence_ength,dim]), dtype=tf.float32)

def Model(features, labels, mode, params):
    TRAIN = mode == tf.estimator.ModeKeys.TRAIN
    EVAL = mode == tf.estimator.ModeKeys.EVAL
    pREDICT = mode == tf.estimator.ModeKeys.PREDICT

    position_encode = positional_encoding(params['embedding_size'], params['max_sequence_length'])
    if params['xavier_initializer'] :
        embedding_initializer = 'glorot_normal'
    else :
        embedding_initializer = 'uniform'

    embedding = tf.keras.layers.Embedding(params['vocabulary_length'], params['embedding_size'], embeddings_initializer=embedding_initializer)
    x_embedded_matrix = embedding(featrues['input']) + position_encode
    y_embedded_matrix = embedding(features['output']) + position_encode

    encoder_outputs = encoder(x_embedded_matrix, params['model_hidden_size'], params['ffn_hidden_size'], params['attention_head_size'], params['layer_size'])
    decoder_outputs = decoder(y_embedded_matrix, encoder_outputs, params['model_hidden_size'], params['ffn_hidden_size'], params['attention_head_size'], params['layer_size'])
    logits = tf.keras.layers.Dense(params['vocabulary_length'])(decoder_outputs)
    predict = tf.argmax(logits, 2)
