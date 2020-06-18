import tensorflow as tf
import numpy as np
from word_id_test import Word_Id_Map
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_options)

# with tf.device('/cpu:0'):
batch_size = 1
sequence_length = 10
num_encoder_symbols = 1004
num_decoder_symbols = 1004
embedding_size = 256
hidden_size = 256
num_layers = 2

encoder_inputs = tf.placeholder(dtype=tf.int32, shape=[batch_size, sequence_length])
decoder_inputs = tf.placeholder(dtype=tf.int32, shape=[batch_size, sequence_length])

cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)

results, states = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
    tf.unstack(encoder_inputs, axis=1),
    tf.unstack(decoder_inputs, axis=1),
    cell,
    num_encoder_symbols,
    num_decoder_symbols,
    embedding_size,
    feed_previous=True,
)
logits = tf.stack(results, axis=1)
pred = tf.argmax(logits, axis=2)

saver = tf.train.Saver()
# sess = tf.Session()#cpu
sess = tf.Session(config=config)#gpu

module_file = tf.train.latest_checkpoint('./model/')
saver.restore(sess, module_file)
map1 = Word_Id_Map()
# encoder_input = map.sentence2ids(['you', 'want', 'to', 'turn', 'twitter', 'followers', 'into', 'blog', 'readers'])


def pred_f(Sess,encod_inputs, decod_inputs,map1,pred,recive_c=' '):
    sess = Sess
    recive_c = recive_c.lower().strip().split()
    encoder_input = map1.sentence2ids(recive_c)
    if isinstance(encoder_input, list):
        encoder_input = encoder_input + [3 for i in range(0, 10 - len(encoder_input))]
        encoder_input = np.asarray([np.asarray(encoder_input)])
        decoder_input = np.zeros([1, 10])
        # print('encoder_input : ', encoder_input)
        # print('decoder_input : ', decoder_input)
        pred_value = sess.run(pred, feed_dict={encod_inputs: encoder_input, decod_inputs: decoder_input})
        # print(pred_value)
        sentence = map1.ids2sentence(pred_value[0])
        for i in range(len(sentence)):
            if sentence[i] == '<eos>' or sentence[i] == '<pad>':
                break
            try:
                cc = cc + ' ' + sentence[i]
            except:
                cc = sentence[i]
    elif isinstance(encoder_input, str):
        cc = encoder_input
    #print sentence
    return cc
    #return 'qwe'

#reply_content = pred_f(sess,recive_c='we love you')
#print(reply_content)
