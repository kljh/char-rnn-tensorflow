import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq
import numpy as np
import os, json

class Model():
    def __init__(self, args, training=True):
        self.args = args
        if not training:
            args.batch_size = 1
            args.seq_length = 1

        # choose different rnn cell 
        if args.model == 'rnn':
            cell_fn = rnn.RNNCell
        elif args.model == 'gru':
            cell_fn = rnn.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn.LSTMCell
        elif args.model == 'nas':
            cell_fn = rnn.NASCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        # warp multi layered rnn cell into one cell with dropout
        cells = []
        for _ in range(args.num_layers):
            cell = cell_fn(args.rnn_size)
            if training and (args.output_keep_prob < 1.0 or args.input_keep_prob < 1.0):
                cell = rnn.DropoutWrapper(cell,
                                          input_keep_prob=args.input_keep_prob,
                                          output_keep_prob=args.output_keep_prob)
            cells.append(cell)
        self.cell = cell = rnn.MultiRNNCell(cells, state_is_tuple=True)

        # input/target data (int32 since input is char-level)
        self.input_data = tf.placeholder(
            tf.int32, [args.batch_size, args.seq_length], name="input_data")
        self.targets = tf.placeholder(
            tf.int32, [args.batch_size, args.seq_length], name="target_data")
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)

        # softmax output layer, use softmax to classify
        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable("softmax_w",
                                        [args.rnn_size, args.vocab_size])
            softmax_b = tf.get_variable("softmax_b", [args.vocab_size])

        with tf.variable_scope('rnnlm_1'):
            with tf.variable_scope('rnnlm'):
                with tf.variable_scope('multi_rnn_cell'):
                    with tf.variable_scope('cell_0'):
                        with tf.variable_scope('cell_0'):
                            with tf.variable_scope('lstm_cell'):
                                with tf.variable_scope('add'):
                                    y = tf.get_variable("y", [args.rnn_size])

        self.softmax_w = softmax_w
        self.softmax_b = softmax_b
        self.wtfy = y
        
        # transform input to embedding
        embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size])
        inputsEmbedded = tf.nn.embedding_lookup(embedding, self.input_data)

        self.embedding = embedding
        
        # dropout beta testing: double check which one should affect next line
        if training and args.output_keep_prob:
            inputsEmbedded = tf.nn.dropout(inputsEmbedded, args.output_keep_prob)

        # unstack the input to fits in rnn model
        inputs = tf.split(inputsEmbedded, args.seq_length, 1)
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        self.inputsEmbedded = inputsEmbedded
        self.inputs = inputs

        # loop function for rnn_decoder, which take the previous i-th cell's output and generate the (i+1)-th cell's input
        def loop(prev, _):
            prev = tf.matmul(prev, softmax_w) + softmax_b
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(embedding, prev_symbol)

        # rnn_decoder to generate the ouputs and final state. When we are not training the model, we use the loop function.
        outputs, last_state = legacy_seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=loop if not training else None, scope='rnnlm')
        output = tf.reshape(tf.concat(outputs, 1), [-1, args.rnn_size])

        # output layer
        self.logits = tf.matmul(output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)

        # loss is calculate by the log loss and taking the average.
        loss = legacy_seq2seq.sequence_loss_by_example(
                [self.logits],
                [tf.reshape(self.targets, [-1])],
                [tf.ones([args.batch_size * args.seq_length])])
        with tf.name_scope('cost'):
            self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length
        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()

        # calculate gradients
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                args.grad_clip)
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.lr)

        # apply gradient change to the all the trainable variable.
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        # instrument tensorboard
        tf.summary.histogram('logits', self.logits)
        tf.summary.histogram('loss', loss)
        tf.summary.scalar('train_loss', self.cost)

    def sample(self, sess, chars, vocab, num=200, prime='The ', sampling_type=1):
        #print("prime", prime)
        
        state = sess.run(self.cell.zero_state(1, tf.float32))
        """
        for char in prime[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state: state}
            [state] = sess.run([self.final_state], feed)
        """

        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        ret = prime
        char = prime[-1]
        for it in range(num):
            #print("char", char, vocab[char])
            
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state: state}
            
            # to augment
            [probs, state, init_state, ine, ins, varis, softmax_w, softmax_b, wtfy, embedding] = sess.run([self.probs, self.final_state, self.initial_state, self.inputsEmbedded, self.inputs, self.cell.variables, self.softmax_w, self.softmax_b, self.wtfy, self.embedding ], feed) # weight
            p = probs[0]
            
            #print("p", probs)
            #print("s", probs)
            #print("bias0", bias0)
            if it < 3 :
                path = os.path.join("save", 'js', '_iter'+str(it)+'_input_embedded.json')
                with open(path, 'w') as fo:
                    #json.dump([x*1 for x in list(bias0) ], fo)
                    json.dump(ine.tolist(), fo)
            
                path = os.path.join("save", 'js', '_iter'+str(it)+'_input_squeezed.json')
                with open(path, 'w') as fo:
                    #json.dump([x*1 for x in list(bias0) ], fo)
                    json.dump([x.tolist() for x in ins], fo)
                
                path = os.path.join("save", 'js', '_iter'+str(it)+'_input_wtfy.json')
                with open(path, 'w') as fo:
                    #json.dump([x*1 for x in list(bias0) ], fo)
                    json.dump(wtfy.tolist() , fo)
            
            if it == 0 or it == 1 :

                path = os.path.join("save", 'js', '_iter'+str(it)+'__embedding.json')
                with open(path, 'w') as fo:
                    json.dump(embedding.tolist(), fo)
            
                path = os.path.join("save", 'js', '_iter'+str(it)+'__softmax_w.json')
                with open(path, 'w') as fo:
                    json.dump(softmax_w.tolist(), fo)
                
                path = os.path.join("save", 'js', '_iter'+str(it)+'__softmax_b.json')
                with open(path, 'w') as fo:
                    json.dump(softmax_b.tolist(), fo)
                
                for iv, vari in enumerate(varis):
                    path = os.path.join("save", 'js', '_iter'+str(it)+'_variable'+str(iv)+'.json')
                    with open(path, 'w') as fo:
                        #json.dump([x*1 for x in list(bias0) ], fo)
                        json.dump(vari.tolist(), fo)
            
            if it < 3 :
                for iv, stat in enumerate(state):
                    path = os.path.join("save", 'js', '_iter'+str(it)+'_final_state'+str(iv)+'_c.json')
                    with open(path, 'w') as fo:
                        json.dump(stat.c.tolist(), fo, indent=4)
                    path = os.path.join("save", 'js', '_iter'+str(it)+'_final_state'+str(iv)+'_h.json')
                    with open(path, 'w') as fo:
                        json.dump(stat.h.tolist(), fo, indent=4)

            if it < 3 :
                for iv, stat in enumerate(init_state):
                    path = os.path.join("save", 'js', '_iter'+str(it)+'_init_state'+str(iv)+'_c.json')
                    with open(path, 'w') as fo:
                        json.dump(stat.c.tolist(), fo, indent=4)
                    path = os.path.join("save", 'js', '_iter'+str(it)+'_init_state'+str(iv)+'_h.json')
                    with open(path, 'w') as fo:
                        json.dump(stat.h.tolist(), fo, indent=4)
            
            if sampling_type == 0:
                sample = np.argmax(p)
            elif sampling_type == 2:
                if char == ' ':
                    sample = weighted_pick(p)
                else:
                    sample = np.argmax(p)
            else:  # sampling_type == 1 default:
                sample = weighted_pick(p)

            pred = chars[sample]
            ret += pred
            char = pred
        return ret
