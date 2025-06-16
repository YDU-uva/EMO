from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
from tensorflow.contrib.distributions import Normal

from image_embedding import ImageEmbedding
from metadag import MetaGraph
from task_embedding import LSTMAutoencoder
from utils import mse, xent, conv_block, normalize

FLAGS = flags.FLAGS


class EMO:

    
    def __init__(self, sess, dim_input=1, dim_output=1, test_num_updates=5):
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.update_lr = FLAGS.update_lr
        self.meta_lr = tf.placeholder_with_default(FLAGS.meta_lr, ())
        self.classification = False
        self.test_num_updates = test_num_updates
        self.sess = sess
        

        self.num_models = 5  
        self.uncertainty_weight = 0.1  
        self.expected_output_weight = 1.0  
        

        self.lstmae = LSTMAutoencoder(hidden_num=FLAGS.hidden_dim, 
                                      input_num=FLAGS.hidden_dim + FLAGS.num_classes,
                                      name='lstm_ae')
        self.lstmae_graph = LSTMAutoencoder(hidden_num=FLAGS.hidden_dim, 
                                            input_num=FLAGS.hidden_dim,
                                            name='lstm_ae_graph')
        

        if FLAGS.datasource in ['2D']:
            self.metagraph = MetaGraph(input_dim=FLAGS.sync_filters, hidden_dim=FLAGS.sync_filters)
        elif FLAGS.datasource in ['plainmulti', 'artmulti']:
            self.metagraph = MetaGraph(input_dim=FLAGS.hidden_dim, hidden_dim=FLAGS.hidden_dim)


        if FLAGS.datasource in ['2D']:
            self.dim_hidden = [40, 40]
            self.loss_func = mse
            self.forward = self.forward_fc
            self.construct_weights = self.construct_fc_weights
        elif FLAGS.datasource in ['plainmulti', 'artmulti']:
            self.loss_func = xent
            self.classification = True
            self.dim_hidden = FLAGS.num_filters
            self.forward = self.forward_conv
            self.construct_weights = self.construct_conv_weights
            self.channels = 3
            self.img_size = int(np.sqrt(self.dim_input / self.channels))
            self.image_embed = ImageEmbedding(hidden_num=FLAGS.task_embedding_num_filters, 
                                              channels=self.channels,
                                              conv_initializer=tf.truncated_normal_initializer(stddev=0.04))
        else:
            raise ValueError('Unrecognized data source.')

    def compute_expected_output(self, outputs_list):
        """计算期望输出"""
        expected_output = tf.reduce_mean(tf.stack(outputs_list), axis=0)
        return expected_output
    
    def compute_uncertainty(self, outputs_list):
        """计算预测不确定性"""
        outputs_stack = tf.stack(outputs_list)
        mean_output = tf.reduce_mean(outputs_stack, axis=0)
        variance = tf.reduce_mean(tf.square(outputs_stack - mean_output), axis=0)
        uncertainty = tf.reduce_mean(variance)
        return uncertainty

    def construct_model(self, input_tensors=None, prefix='metatrain_'):
        if input_tensors is None:
            self.inputa = tf.placeholder(tf.float32, shape=(FLAGS.meta_batch_size, FLAGS.update_batch_size, 2))
            self.inputb = tf.placeholder(tf.float32,
                                         shape=(FLAGS.meta_batch_size, FLAGS.update_batch_size_eval, 2))
            self.labela = tf.placeholder(tf.float32, shape=(FLAGS.meta_batch_size, FLAGS.update_batch_size, 1))
            self.labelb = tf.placeholder(tf.float32,
                                         shape=(FLAGS.meta_batch_size, FLAGS.update_batch_size_eval, 1))
        else:
            self.inputa = input_tensors['inputa']
            self.inputb = input_tensors['inputb']
            self.labela = input_tensors['labela']
            self.labelb = input_tensors['labelb']

        with tf.variable_scope('emo_model', reuse=tf.AUTO_REUSE) as training_scope:
            if 'weights' in dir(self):
                training_scope.reuse_variables()
                weights = self.weights
            else:
                # 构建多个模型的权重
                self.weights = weights = [self.construct_weights() for _ in range(self.num_models)]

            num_updates = max(self.test_num_updates, FLAGS.num_updates)

            def task_metalearn(inp, reuse=True):
                inputa, inputb, labela, labelb = inp
                
                # 任务嵌入
                if FLAGS.datasource in ['2D']:
                    input_task_emb = tf.concat((inputa, labela), axis=-1)
                    with tf.variable_scope('first_embedding_sync', reuse=tf.AUTO_REUSE):
                        input_task_emb = tf.layers.dense(input_task_emb, units=FLAGS.sync_filters,
                                                         name='first_embedding_sync_dense')
                    if FLAGS.num_classes < FLAGS.update_batch_size:
                        with tf.variable_scope('reg_clustering', reuse=tf.AUTO_REUSE):
                            assign_mat = tf.nn.softmax(tf.layers.dense(input_task_emb, units=FLAGS.num_classes), dim=1)
                            input_task_emb_cat = tf.matmul(tf.transpose(assign_mat, perm=[1, 0]), input_task_emb)

                elif FLAGS.datasource in ['plainmulti', 'artmulti']:
                    input_task_emb = self.image_embed.model(tf.reshape(inputa,
                                                                       [-1, self.img_size, self.img_size,
                                                                        self.channels]))

                    proto_emb = []
                    labela2idx = tf.argmax(labela, axis=1)
                    for class_idx in range(FLAGS.num_classes):
                        tmp_gs = tf.equal(labela2idx, class_idx)
                        gs = tf.where(tmp_gs)
                        new_vec = tf.reduce_mean(tf.gather(input_task_emb, gs), axis=0)
                        proto_emb.append(new_vec)
                    proto_emb = tf.squeeze(tf.stack(proto_emb))

                    label_cat = tf.eye(5)
                    input_task_emb_cat = tf.concat((proto_emb, label_cat), axis=-1)

                # 任务嵌入处理
                if FLAGS.datasource in ['2D']:
                    task_embed_vec, task_emb_loss = self.lstmae.model(input_task_emb)
                    propagate_knowledge = self.metagraph.model(input_task_emb_cat)
                elif FLAGS.datasource in ['plainmulti', 'artmulti']:
                    task_embed_vec, task_emb_loss = self.lstmae.model(input_task_emb_cat)
                    propagate_knowledge = self.metagraph.model(proto_emb)

                task_embed_vec_graph, task_emb_loss_graph = self.lstmae_graph.model(propagate_knowledge)
                task_enhanced_emb_vec = tf.concat([task_embed_vec, task_embed_vec_graph], axis=1)

                # 为每个集成模型生成任务特定权重
                task_weights_list = []
                for model_idx in range(self.num_models):
                    with tf.variable_scope(f'task_specific_mapping_{model_idx}', reuse=tf.AUTO_REUSE):
                        eta = []
                        for key in weights[model_idx].keys():
                            weight_size = np.prod(weights[model_idx][key].get_shape().as_list())
                            eta.append(tf.reshape(
                                tf.layers.dense(task_enhanced_emb_vec, weight_size, activation=tf.nn.sigmoid,
                                                name=f'eta_{key}_model_{model_idx}'), 
                                tf.shape(weights[model_idx][key])))
                        eta = dict(zip(weights[model_idx].keys(), eta))
                        task_weights = dict(zip(weights[model_idx].keys(), 
                                                [weights[model_idx][key] * eta[key] 
                                                 for key in weights[model_idx].keys()]))
                        task_weights_list.append(task_weights)


                task_outputbs_ensemble = [[] for _ in range(self.num_models)]
                task_lossesb_ensemble = [[] for _ in range(self.num_models)]
                task_expected_outputs = []
                task_uncertainties = []

                if self.classification:
                    task_accuraciesb_ensemble = [[] for _ in range(self.num_models)]


                task_outputas = []
                task_lossesas = []
                for model_idx in range(self.num_models):
                    task_outputa = self.forward(inputa, task_weights_list[model_idx], reuse=reuse)
                    task_lossa = self.loss_func(task_outputa, labela)
                    task_outputas.append(task_outputa)
                    task_lossesas.append(task_lossa)


                expected_outputa = self.compute_expected_output(task_outputas)
                uncertainty_a = self.compute_uncertainty(task_outputas)


                fast_weights_list = []
                for model_idx in range(self.num_models):
                    grads = tf.gradients(task_lossesas[model_idx], list(task_weights_list[model_idx].values()))
                    if FLAGS.stop_grad:
                        grads = [tf.stop_gradient(grad) for grad in grads]
                    gradients = dict(zip(task_weights_list[model_idx].keys(), grads))
                    fast_weights = dict(
                        zip(task_weights_list[model_idx].keys(),
                            [task_weights_list[model_idx][key] - self.update_lr * gradients[key] 
                             for key in task_weights_list[model_idx].keys()]))
                    fast_weights_list.append(fast_weights)


                outputbs = []
                for model_idx in range(self.num_models):
                    output = self.forward(inputb, fast_weights_list[model_idx], reuse=True)
                    outputbs.append(output)
                    task_outputbs_ensemble[model_idx].append(output)
                    task_lossesb_ensemble[model_idx].append(self.loss_func(output, labelb))

                expected_outputb = self.compute_expected_output(outputbs)
                uncertainty_b = self.compute_uncertainty(outputbs)
                task_expected_outputs.append(expected_outputb)
                task_uncertainties.append(uncertainty_b)


                for j in range(num_updates - 1):
                    for model_idx in range(self.num_models):
                        loss = self.loss_func(self.forward(inputa, fast_weights_list[model_idx], reuse=True), labela)
                        grads = tf.gradients(loss, list(fast_weights_list[model_idx].values()))
                        if FLAGS.stop_grad:
                            grads = [tf.stop_gradient(grad) for grad in grads]
                        gradients = dict(zip(fast_weights_list[model_idx].keys(), grads))
                        fast_weights_list[model_idx] = dict(
                            zip(fast_weights_list[model_idx].keys(),
                                [fast_weights_list[model_idx][key] - self.update_lr * gradients[key] 
                                 for key in fast_weights_list[model_idx].keys()]))
                    

                    outputbs = []
                    for model_idx in range(self.num_models):
                        output = self.forward(inputb, fast_weights_list[model_idx], reuse=True)
                        outputbs.append(output)
                        task_outputbs_ensemble[model_idx].append(output)
                        task_lossesb_ensemble[model_idx].append(self.loss_func(output, labelb))
                    
                    expected_outputb = self.compute_expected_output(outputbs)
                    uncertainty_b = self.compute_uncertainty(outputbs)
                    task_expected_outputs.append(expected_outputb)
                    task_uncertainties.append(uncertainty_b)


                task_output = [task_emb_loss, task_emb_loss_graph, expected_outputa, task_expected_outputs, 
                              tf.reduce_mean(task_lossesas), 
                              [self.loss_func(task_expected_outputs[j], labelb) for j in range(num_updates)],
                              uncertainty_a, task_uncertainties]

                if self.classification:
                    task_accuracya = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(expected_outputa), 1),
                                                                 tf.argmax(labela, 1))
                    task_accuraciesb = []
                    for j in range(num_updates):
                        task_accuraciesb.append(
                            tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_expected_outputs[j]), 1),
                                                        tf.argmax(labelb, 1)))
                    task_output.extend([task_accuracya, task_accuraciesb])

                return task_output

            if FLAGS.norm != 'None':
                unused = task_metalearn((self.inputa[0], self.inputb[0], self.labela[0], self.labelb[0]), False)

            out_dtype = [tf.float32, tf.float32, tf.float32, [tf.float32] * num_updates, tf.float32,
                         [tf.float32] * num_updates, tf.float32, [tf.float32] * num_updates]
            if self.classification:
                out_dtype.extend([tf.float32, [tf.float32] * num_updates])
                
            result = tf.map_fn(task_metalearn, elems=(self.inputa, self.inputb, self.labela, self.labelb),
                               dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)
            
            if self.classification:
                (emb_loss, emb_loss_graph, outputas, outputbs, lossesa, lossesb, 
                 uncertainty_a, uncertainty_b, accuraciesa, accuraciesb) = result
            else:
                (emb_loss, emb_loss_graph, outputas, outputbs, lossesa, lossesb, 
                 uncertainty_a, uncertainty_b) = result


        if 'train' in prefix:
            self.total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(FLAGS.meta_batch_size)
            self.total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size) 
                                                  for j in range(num_updates)]
            self.total_embed_loss = tf.reduce_sum(emb_loss) / tf.to_float(FLAGS.meta_batch_size)
            self.total_embed_loss_graph = tf.reduce_sum(emb_loss_graph) / tf.to_float(FLAGS.meta_batch_size)
            

            self.total_uncertainty_loss = tf.reduce_sum(uncertainty_a) / tf.to_float(FLAGS.meta_batch_size)
            self.total_uncertainty_losses2 = [tf.reduce_sum(uncertainty_b[j]) / tf.to_float(FLAGS.meta_batch_size) 
                                               for j in range(num_updates)]
            

            self.emo_loss = (self.expected_output_weight * total_losses2[-1] + 
                             self.uncertainty_weight * self.total_uncertainty_losses2[-1])
            
            self.outputas, self.outputbs = outputas, outputbs
            
            if self.classification:
                self.total_accuracy1 = tf.reduce_sum(accuraciesa) / tf.to_float(FLAGS.meta_batch_size)
                self.total_accuracies2 = [
                    tf.reduce_sum(accuraciesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
            

            self.metatrain_op = tf.train.AdamOptimizer(self.meta_lr).minimize(self.emo_loss)
            self.pretrain_op = tf.train.AdamOptimizer(self.meta_lr).minimize(total_loss1)

        else:  
            self.metaval_total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(FLAGS.meta_batch_size)
            self.metaval_total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size) 
                                          for j in range(num_updates)]
            self.metaval_total_uncertainty_loss = tf.reduce_sum(uncertainty_a) / tf.to_float(FLAGS.meta_batch_size)
            self.metaval_total_uncertainty_losses2 = [tf.reduce_sum(uncertainty_b[j]) / tf.to_float(FLAGS.meta_batch_size) 
                                                      for j in range(num_updates)]
            
            if self.classification:
                self.metaval_total_accuracy1 = tf.reduce_sum(accuraciesa) / tf.to_float(FLAGS.meta_batch_size)
                self.metaval_total_accuracies2 = [
                    tf.reduce_sum(accuraciesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]


        if 'train' in prefix:
            tf.summary.scalar(prefix + 'Pre-update loss', self.total_loss1)
            tf.summary.scalar(prefix + 'Post-update loss', self.total_losses2[-1])
            tf.summary.scalar(prefix + 'EMO loss', self.emo_loss)
            tf.summary.scalar(prefix + 'Uncertainty loss', self.total_uncertainty_losses2[-1])

    def construct_fc_weights(self):
        weights = {}
        weights['w1'] = tf.Variable(tf.truncated_normal([self.dim_input, self.dim_hidden[0]], stddev=0.01))
        weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden[0]]))
        for i in range(1, len(self.dim_hidden)):
            weights['w' + str(i + 1)] = tf.Variable(
                tf.truncated_normal([self.dim_hidden[i - 1], self.dim_hidden[i]], stddev=0.01))
            weights['b' + str(i + 1)] = tf.Variable(tf.zeros([self.dim_hidden[i]]))
        weights['w' + str(len(self.dim_hidden) + 1)] = tf.Variable(
            tf.truncated_normal([self.dim_hidden[-1], self.dim_output], stddev=0.01))
        weights['b' + str(len(self.dim_hidden) + 1)] = tf.Variable(tf.zeros([self.dim_output]))
        return weights

    def forward_fc(self, inp, weights, reuse=False):
        hidden = normalize(tf.matmul(inp, weights['w1']) + weights['b1'], activation=tf.nn.relu, reuse=reuse,
                           scope='0')
        for i in range(1, len(self.dim_hidden)):
            hidden = normalize(tf.matmul(hidden, weights['w' + str(i + 1)]) + weights['b' + str(i + 1)],
                               activation=tf.nn.relu, reuse=reuse, scope=str(i + 1))
        return tf.matmul(hidden, weights['w' + str(len(self.dim_hidden) + 1)]) + weights[
            'b' + str(len(self.dim_hidden) + 1)]

    def construct_conv_weights(self):
        weights = {}

        dtype = tf.float32
        conv_initializer = tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
        fc_initializer = tf.contrib.layers.xavier_initializer(dtype=dtype)
        k = 3

        weights['conv1'] = tf.get_variable('conv1', [k, k, self.channels, self.dim_hidden],
                                           initializer=conv_initializer, dtype=dtype)
        weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden]))
        weights['conv2'] = tf.get_variable('conv2', [k, k, self.dim_hidden, self.dim_hidden],
                                           initializer=conv_initializer, dtype=dtype)
        weights['b2'] = tf.Variable(tf.zeros([self.dim_hidden]))
        weights['conv3'] = tf.get_variable('conv3', [k, k, self.dim_hidden, self.dim_hidden],
                                           initializer=conv_initializer, dtype=dtype)
        weights['b3'] = tf.Variable(tf.zeros([self.dim_hidden]))
        weights['conv4'] = tf.get_variable('conv4', [k, k, self.dim_hidden, self.dim_hidden],
                                           initializer=conv_initializer, dtype=dtype)
        weights['b4'] = tf.Variable(tf.zeros([self.dim_hidden]))
        if FLAGS.datasource == 'plainmulti':
            weights['w5'] = tf.get_variable('w5', [self.dim_hidden, self.dim_output], initializer=fc_initializer)
            weights['b5'] = tf.Variable(tf.zeros([self.dim_output]), name='b5')
        elif FLAGS.datasource == 'artmulti':
            weights['w5'] = tf.get_variable('w5', [self.dim_hidden * 5 * 5, self.dim_output], initializer=fc_initializer)
            weights['b5'] = tf.Variable(tf.zeros([self.dim_output]), name='b5')

        return weights

    def forward_conv(self, inp, weights, reuse=False, scope=''):
        channels = self.channels
        inp = tf.reshape(inp, [-1, self.img_size, self.img_size, channels])

        hidden1 = conv_block(inp, weights['conv1'], weights['b1'], reuse, scope + '0')
        hidden2 = conv_block(hidden1, weights['conv2'], weights['b2'], reuse, scope + '1')
        hidden3 = conv_block(hidden2, weights['conv3'], weights['b3'], reuse, scope + '2')
        hidden4 = conv_block(hidden3, weights['conv4'], weights['b4'], reuse, scope + '3')
        if FLAGS.datasource == 'plainmulti':
            hidden4 = tf.reduce_mean(hidden4, [1, 2])
        elif FLAGS.datasource == 'artmulti':
            hidden4 = tf.reshape(hidden4, [-1, np.prod([int(dim) for dim in hidden4.get_shape()[1:]])])

        return tf.matmul(hidden4, weights['w5']) + weights['b5'] 