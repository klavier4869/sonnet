''' predicted sin wave using one before data using dnn. '''
import tensorflow as tf
import sonnet as snt

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sinwave_dataset

def initArgParser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--max_steps', type=int, default=5000,
                      help='Number of steps to run trainer.')
  parser.add_argument('--val_init_stddev', type=int, default=0.1,
                      help='Number of variable initializers stddev.')
  parser.add_argument('--val_reg_scale', type=int, default=0.1,
                      help='Number of variable l1 l2 scale.')
  parser.add_argument('--hidden_size', type=int, default=1,
                      help='Number of hidden layers.')
  parser.add_argument('--unit_size', type=int, default=10,
                      help='Number of unit size.')
  parser.add_argument('--batch_size', type=float, default=50,
                      help='Number of batch size.')
  parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Initial learning rate')
  parser.add_argument('--dropout', type=float, default=0.9,
                      help='Keep probability for training dropout.')
  parser.add_argument('--log_dir', type=str, default='data',
                      help='Summaries log directory')
  return parser

def build(inputs, keep_prob):
  initializers = {'w': tf.truncated_normal_initializer(
                        stddev=FLAGS.val_init_stddev),
                  'b': tf.truncated_normal_initializer(
                        stddev=FLAGS.val_init_stddev)}
  regularizers = {'w': tf.contrib.layers.l1_regularizer(
                        scale=FLAGS.val_reg_scale),
                  'b': tf.contrib.layers.l2_regularizer(
                        scale=FLAGS.val_reg_scale)}

  if(FLAGS.hidden_size == 1):
    outputs = snt.Linear(output_size=1,
                         initializers=initializers,
                         regularizers=regularizers)(inputs)
    return outputs

  outputs = snt.Linear(output_size=FLAGS.unit_size,
                       initializers=initializers,
                       regularizers=regularizers)(inputs)
  outputs = tf.nn.relu(outputs)
  #outputs = tf.nn.dropout(outputs, keep_prob)
  for _ in range(FLAGS.hidden_size-2):
    outputs = snt.Linear(output_size=FLAGS.unit_size,
                         initializers=initializers,
                         regularizers=regularizers)(outputs)
    outputs = tf.nn.relu(outputs)
  #outputs = tf.nn.dropout(outputs, keep_prob)
  outputs = snt.Linear(output_size=1,
                       initializers=initializers,
                       regularizers=regularizers)(outputs)
  return outputs

def train():
  # build graph
  with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 1], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 1], name='y-input')
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
  model = snt.Module(build)
  y = model(x, keep_prob)

  with tf.name_scope('loss'):
    diff = tf.nn.l2_loss(y_ - y)
    with tf.name_scope('total'):
      l2 = tf.reduce_mean(diff)
    with tf.name_scope('weight_decay'):
      tf.add_to_collection('losses', l2)
      loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    with tf.name_scope('total_regularization_loss'):
      graph_regularizers = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
      total_regularization_loss = tf.reduce_sum(graph_regularizers)
  tf.summary.scalar('loss', loss)
  tf.summary.scalar('total_regularization_loss', total_regularization_loss)

  with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
      loss + total_regularization_loss)

  merged = tf.summary.merge_all()
  sess = tf.InteractiveSession()
  train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/tb/train', sess.graph)
  test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/tb/test')
  tf.global_variables_initializer().run()

  # train
  dataset = sinwave_dataset.SinDataset(batch_size=FLAGS.batch_size)
  def feed_dict(train):
    '''Make a TensorFlow feed_dict: maps data onto Tensor placeholders.'''
    if train:
        xs, ys = dataset.fetch_train()
        k = FLAGS.dropout
    else:
        xs, ys = dataset.fetch_test()
        k = 1.0
    return {x: xs, y_: ys, keep_prob: k}

  plotdata = dataset.all_data
  sin_t = plotdata['sin_t'].values.reshape(-1, 1)
  def write_log(cur_step):
    plotdata['predicted_use_real'] = sess.run(y, {x: sin_t, keep_prob: 1.0})

    predicted_sin = [0.0]
    for i in range(1, 4000):
      inputs = np.array(predicted_sin[i-1]).reshape(-1, 1)
      sin_i = sess.run(y, {x: inputs, keep_prob: 1.0})[0]
      predicted_sin.append(sin_i[0])

    plt.figure()
    plt.plot(range(2000), predicted_sin[:2000])
    plt.plot(range(2000, 4000), predicted_sin[2000:4000])
    plotdata['predicted_use_real'].plot()
    plotdata['sin_t'].plot(style='k--')
    plt.xlabel('t', fontsize=12, fontname='serif')
    plt.ylabel('sin(t)', fontsize=12, fontname='serif')
    plt.legend(('predict(train)', 'predict(test)','predict(real)', 'real'))
    plt.savefig(FLAGS.log_dir + '/svg/' + str(cur_step) + '.svg')

    plt.figure()
    for i in range(2):
      if(i == 0):
          start = 0
          end   = 2000
      else:
          start = 2000
          end   = 4000
      predicted_y  = np.array(predicted_sin[start:end])
      real_y = plotdata['sin_t'][start:end].values
      abs_loss = abs(predicted_y - real_y)
      abs_loss[np.isnan(abs_loss)] = 0
      plt.hist(abs_loss, bins=16, alpha=0.3)
    plt.xlabel('loss', fontsize=12, fontname='serif')
    plt.ylabel('num', fontsize=12, fontname='serif')
    plt.legend(('loss(train)','loss(test)'))
    plt.savefig(FLAGS.log_dir + '/svg/loss' + str(cur_step) + '.svg')

  for i in range(FLAGS.max_steps):
    summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
    train_writer.add_summary(summary, i)
    if i % 1000 == 999:
      write_log(i+1)
      print('step', i+1)

  saver = tf.train.Saver()
  saver.save(sess, FLAGS.log_dir + '/model/model.ckpt')

def main(_):
  # init tensor_borad_logdir
  tensor_borad_logdir = FLAGS.log_dir + '/tb'
  if tf.gfile.Exists(tensor_borad_logdir):
    tf.gfile.DeleteRecursively(tensor_borad_logdir)
  tf.gfile.MakeDirs(tensor_borad_logdir)
  # init model_logdir
  model_logdir = FLAGS.log_dir + '/model'
  if not tf.gfile.Exists(model_logdir):
    tf.gfile.MakeDirs(model_logdir)
  # init svg_logdir
  svg_logdir = FLAGS.log_dir + '/svg'
  if not tf.gfile.Exists(svg_logdir):
    tf.gfile.MakeDirs(svg_logdir)
  train()

if __name__ == '__main__':
  parser = initArgParser()
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main)
