#!/usr/bin/env python
#coding=utf-8
# @file  : attenNet
# @time  : 6/8/2020 10:38 PM
# @author: shishishu

import tensorflow as tf
from lib.utils.generalNet import GeneralNet


class MultiHeadAtten:

    def __init__(self, num_atten_head, d_model):
        assert d_model % num_atten_head == 0, 'wrong setting in num of attention head'
        self.d_model = d_model
        self.head = num_atten_head  # AT
        self.d_k = d_model // num_atten_head  # K, # assume d_v = d_k always
        # self.p_atten = None

    def multi_atten(self, inputs, activation, keep_prob, scope_name='mha'):
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
            batch_size = tf.shape(inputs)[0]  # [B, M, Last]
            qkv = list(map(lambda x: tf.layers.dense(x, units=self.d_model), (inputs, inputs, inputs)))  # [B, M, d_model]
            # [B, M, d_model] -> reshape: [B, M, AT, K] -> transpose: [B, AT, M, K]
            query, key, value = list(
                map(lambda x: tf.transpose(tf.reshape(x, shape=[batch_size, -1, self.head, self.d_k]), perm=[0, 2, 1, 3]),
                    qkv))
            atten_outputs, _ = MultiHeadAtten.atten_net(query, key, value, True)
            # [B, AT, M, K] -> transpose: [B, M, AT, K] -> reshape: [B, M, AT*K]
            atten_outputs = tf.reshape(tf.transpose(atten_outputs, perm=[0, 2, 1, 3]),
                                       shape=[batch_size, -1, self.head * self.d_k])
            acti_func = GeneralNet.get_activation(activation)
            if acti_func:
                atten_outputs = acti_func(atten_outputs)
        return tf.nn.dropout(atten_outputs, keep_prob, name=scope.name)

    def multi_atten_layers(self, inputs, num_atten_layers, activation, keep_prob, scope_name='mha'):
        tmp_inputs = inputs
        for idx in range(num_atten_layers):
            tmp_inputs = self.multi_atten(tmp_inputs, activation, keep_prob, scope_name + '_' + str(idx))
        return tmp_inputs

    @staticmethod
    def atten_net(query, key, value, mask):
        d_k = tf.shape(query)[-1]  # q, k, v: [B, AT, M, K]
        scale = tf.sqrt(tf.cast(d_k, tf.float32))  # a scalar
        # [B, AT, M, K] * [B, AT, K, M] = [B, AT, M, M]
        scores = tf.matmul(query, tf.transpose(key, perm=[0, 1, 3, 2])) / scale
        if mask:
            scores = MultiHeadAtten.atten_mask(scores)  # replace 0 with -1e9
        p_atten = tf.nn.softmax(scores, axis=-1)  # [B, AT, M, M]
        return tf.matmul(p_atten, value), p_atten  # output: [B, AT, M, K]

    @staticmethod
    def atten_mask(scores, force_val=-1e9):
        masker = tf.equal(scores, 0)
        forcer = tf.multiply(tf.ones_like(scores, dtype=tf.float32), force_val)
        return tf.where(masker, forcer, scores)