#!/usr/bin/env python

from __future__ import print_function

import argparse
import os
from six.moves import cPickle

from six import text_type

import time
import json




parser = argparse.ArgumentParser(
                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--save_dir', type=str, default='save/js',
                    help='model directory to store checkpointed models')
parser.add_argument('-n', type=int, default=500,
                    help='number of characters to sample')
parser.add_argument('--prime', type=text_type, default=u'',
                    help='prime text')
parser.add_argument('--sample', type=int, default=1,
                    help='0 to use max at each timestep, 1 to sample at '
                         'each timestep, 2 to sample on spaces')

args = parser.parse_args()

import tensorflow as tf
from model import Model

def sample(args):
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
        #with open(os.path.join(args.save_dir, '_config.json'), 'a') as f:
        #json.dump(saved_args, os.path.join(args.save_dir, 'config.json'))
    
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'rb') as f:
        chars, vocab = cPickle.load(f)
        with open(os.path.join(args.save_dir, '_chars_vocab.json'), 'w') as fo:
            json.dump({ "chars": chars, "vocab": vocab }, fo)

    #Use most frequent char if no prime is given
    if args.prime == '':
        args.prime = chars[0]
    model = Model(saved_args, training=False)
    with tf.Session() as sess:
        # instrument for tensorboard
        summaries = tf.summary.merge_all()
        writer = tf.summary.FileWriter(
                os.path.join("logs", "sample"+time.strftime("%Y-%m-%d-%H-%M-%S")))
        writer.add_graph(sess.graph)

        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print(model.sample(sess, chars, vocab, args.n, args.prime,
                               args.sample).encode('utf-8'))

if __name__ == '__main__':
    sample(args)
