import random

import numpy as np
import tensorflow as tf

from emo import EMO

tf.set_random_seed(1234)
from data_generator import DataGenerator
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

## Dataset/method options
flags.DEFINE_string('datasource', 'plainmulti', '2D or plainmulti or artmulti')
flags.DEFINE_integer('test_dataset', -1,
                     'which data to be test, plainmulti: 0-3, artmulti: 0-11, -1: random select')
flags.DEFINE_integer('num_classes', 5, 'number of classes used in classification (e.g. 5-way classification).')
flags.DEFINE_integer('num_test_task', 1000, 'number of test tasks.')
flags.DEFINE_integer('test_epoch', -1, 'test epoch, only work when test start')

## Training options
flags.DEFINE_integer('metatrain_iterations', 15000,
                     'number of metatraining iterations.')  # 15k for omniglot, 50k for sinusoid
flags.DEFINE_integer('meta_batch_size', 25, 'number of tasks sampled per meta-update')
flags.DEFINE_float('meta_lr', 0.001, 'the base learning rate of the generator')
flags.DEFINE_integer('update_batch_size', 5,
                     'number of examples used for inner gradient update (K for K-shot learning).')
flags.DEFINE_integer('update_batch_size_eval', 10,
                     'number of examples used for inner gradient test (K for K-shot learning).')
flags.DEFINE_float('update_lr', 1e-3, 'step size alpha for inner gradient update.')  # 0.1 for omniglot
flags.DEFINE_integer('num_updates', 1, 'number of inner gradient updates during training.')
flags.DEFINE_integer('num_updates_test', 20, 'number of inner gradient updates during training.')
flags.DEFINE_integer('sync_group_num', 6, 'the number of different groups in sync dataset')

## Model options
flags.DEFINE_string('norm', 'batch_norm', 'batch_norm, or None')
flags.DEFINE_integer('hidden_dim', 40, 'output dimension of task embedding')
flags.DEFINE_integer('num_filters', 64, '32 for plainmulti and artmulti')
flags.DEFINE_integer('sync_filters', 40, 'number of dim when combine sync functions.')
flags.DEFINE_bool('stop_grad', False, 'if True, do not use second derivatives in meta-optimization (for speed)')
flags.DEFINE_float('emb_loss_weight', 0.0, 'the weight of autoencoder')
flags.DEFINE_integer('task_embedding_num_filters', 32, 'number of filters for task embedding')
flags.DEFINE_integer('num_vertex', 4, 'number of vertex in the first layer')

## EMO specific options
flags.DEFINE_integer('num_ensemble_models', 5, 'number of ensemble models in EMO')
flags.DEFINE_float('uncertainty_weight', 0.1, 'weight for uncertainty loss')
flags.DEFINE_float('expected_output_weight', 1.0, 'weight for expected output loss')

## Logging, saving, and testing options
flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
flags.DEFINE_string('logdir', './logs_emo/', 'directory for summaries and checkpoints.')
flags.DEFINE_string('datadir', './Data/', 'directory for datasets.')
flags.DEFINE_bool('resume', False, 'resume training if there is a model available')
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_bool('test_set', True, 'Set to true to evaluate on the the test set, False for the validation set.')


def train(model, saver, sess, exp_string, data_generator, resume_itr=0):
    SAVE_INTERVAL = 1000
    if FLAGS.datasource in ['2D']:
        PRINT_INTERVAL = 1000
    else:
        PRINT_INTERVAL = 100

    print('开始EMO模型训练...')

    prelosses, postlosses, embedlosses, emolosses, uncertaintylosses = [], [], [], [], []

    num_classes = data_generator.num_classes

    for itr in range(resume_itr, FLAGS.metatrain_iterations):
        feed_dict = {}
        if FLAGS.datasource == '2D':
            batch_x, batch_y, para_func, sel_set = data_generator.generate_2D_batch()

            inputa = batch_x[:, :num_classes * FLAGS.update_batch_size, :]
            labela = batch_y[:, :num_classes * FLAGS.update_batch_size, :]
            inputb = batch_x[:, num_classes * FLAGS.update_batch_size:, :]
            labelb = batch_y[:, num_classes * FLAGS.update_batch_size:, :]
            feed_dict = {model.inputa: inputa, model.inputb: inputb, model.labela: labela, model.labelb: labelb}

        input_tensors = [model.metatrain_op, model.total_embed_loss, model.total_loss1,
                         model.total_losses2[FLAGS.num_updates - 1], model.emo_loss,
                         model.total_uncertainty_losses2[FLAGS.num_updates - 1]]
        if model.classification:
            input_tensors.extend([model.total_accuracy1, model.total_accuracies2[FLAGS.num_updates - 1]])

        result = sess.run(input_tensors, feed_dict)

        if model.classification:
            _, embed_loss, preloss, postloss, emo_loss, uncertainty_loss, preaccuracy, postaccuracy = result
        else:
            _, embed_loss, preloss, postloss, emo_loss, uncertainty_loss = result

        prelosses.append(preloss)
        postlosses.append(postloss)
        embedlosses.append(embed_loss)
        emolosses.append(emo_loss)
        uncertaintylosses.append(uncertainty_loss)

        if (itr != 0) and itr % PRINT_INTERVAL == 0:
            print_str = f'迭代 {itr}'
            std = np.std(postlosses, 0)
            ci95 = 1.96 * std / np.sqrt(PRINT_INTERVAL)
            print_str += f': 预训练损失: {np.mean(prelosses):.4f}, 后训练损失: {np.mean(postlosses):.4f}'
            print_str += f', 嵌入损失: {np.mean(embedlosses):.4f}, EMO损失: {np.mean(emolosses):.4f}'
            print_str += f', 不确定性损失: {np.mean(uncertaintylosses):.4f}, 置信区间: {ci95:.4f}'

            if model.classification:
                print_str += f', 预训练准确率: {preaccuracy:.4f}, 后训练准确率: {postaccuracy:.4f}'

            print(print_str)
            prelosses, postlosses, embedlosses, emolosses, uncertaintylosses = [], [], [], [], []

        if (itr != 0) and itr % SAVE_INTERVAL == 0:
            saver.save(sess, FLAGS.logdir + '/' + exp_string + '/model' + str(itr))

    saver.save(sess, FLAGS.logdir + '/' + exp_string + '/model' + str(itr))


def test(model, sess, data_generator):
    num_classes = data_generator.num_classes

    metaval_accuracies = []
    metaval_uncertainties = []
    print(f'开始测试，共{FLAGS.num_test_task}个任务')

    for test_itr in range(FLAGS.num_test_task):
        if FLAGS.datasource == '2D':
            batch_x, batch_y, para_func, sel_set = data_generator.generate_2D_batch()

            inputa = batch_x[:, :num_classes * FLAGS.update_batch_size, :]
            inputb = batch_x[:, num_classes * FLAGS.update_batch_size:, :]
            labela = batch_y[:, :num_classes * FLAGS.update_batch_size, :]
            labelb = batch_y[:, num_classes * FLAGS.update_batch_size:, :]

            feed_dict = {model.inputa: inputa, model.inputb: inputb, model.labela: labela, model.labelb: labelb,
                         model.meta_lr: 0.0}
        else:
            feed_dict = {model.meta_lr: 0.0}

        if model.classification:
            result = sess.run([model.metaval_total_accuracy1] + model.metaval_total_accuracies2 +
                              [model.metaval_total_uncertainty_loss] + model.metaval_total_uncertainty_losses2, 
                              feed_dict)
            accuracy_results = result[:len(model.metaval_total_accuracies2) + 1]
            uncertainty_results = result[len(model.metaval_total_accuracies2) + 1:]
            metaval_accuracies.append(accuracy_results)
            metaval_uncertainties.append(uncertainty_results[-1])  # 取最后一步的不确定性
        else:
            result = sess.run([model.metaval_total_loss1] + model.metaval_total_losses2 +
                              [model.metaval_total_uncertainty_loss] + model.metaval_total_uncertainty_losses2, 
                              feed_dict)
            loss_results = result[:len(model.metaval_total_losses2) + 1]
            uncertainty_results = result[len(model.metaval_total_losses2) + 1:]
            metaval_accuracies.append(loss_results)
            metaval_uncertainties.append(uncertainty_results[-1])

    metaval_accuracies = np.array(metaval_accuracies)
    metaval_uncertainties = np.array(metaval_uncertainties)
    
    means = np.mean(metaval_accuracies, 0)
    stds = np.std(metaval_accuracies, 0)
    ci95 = 1.96 * stds / np.sqrt(FLAGS.num_test_task)
    
    uncertainty_mean = np.mean(metaval_uncertainties)
    uncertainty_std = np.std(metaval_uncertainties)

    print('EMO模型测试结果:')
    print('验证准确率/损失的均值, 标准差, 和置信区间:')
    print(f'均值: {means}')
    print(f'标准差: {stds}')
    print(f'置信区间: {ci95}')
    print(f'平均不确定性: {uncertainty_mean:.4f} ± {uncertainty_std:.4f}')


def main():
    sess = tf.InteractiveSession()
    if FLAGS.train:
        test_num_updates = FLAGS.num_updates
    else:
        test_num_updates = FLAGS.num_updates_test

    if FLAGS.train == False:
        orig_meta_batch_size = FLAGS.meta_batch_size
        FLAGS.meta_batch_size = 1

    if FLAGS.datasource in ['2D']:
        data_generator = DataGenerator(FLAGS.update_batch_size + FLAGS.update_batch_size_eval, FLAGS.meta_batch_size)
    else:
        if FLAGS.train:
            data_generator = DataGenerator(FLAGS.update_batch_size + 15,
                                           FLAGS.meta_batch_size)
        else:
            data_generator = DataGenerator(FLAGS.update_batch_size * 2,
                                           FLAGS.meta_batch_size)

    dim_output = data_generator.dim_output
    dim_input = data_generator.dim_input

    if FLAGS.datasource in ['plainmulti', 'artmulti']:
        num_classes = data_generator.num_classes
        if FLAGS.train:
            random.seed(5)
            if FLAGS.datasource == 'plainmulti':
                image_tensor, label_tensor = data_generator.make_data_tensor_plainmulti()
            elif FLAGS.datasource == 'artmulti':
                image_tensor, label_tensor = data_generator.make_data_tensor_artmulti()
            inputa = tf.slice(image_tensor, [0, 0, 0], [-1, num_classes * FLAGS.update_batch_size, -1])
            inputb = tf.slice(image_tensor, [0, num_classes * FLAGS.update_batch_size, 0], [-1, -1, -1])
            labela = tf.slice(label_tensor, [0, 0, 0], [-1, num_classes * FLAGS.update_batch_size, -1])
            labelb = tf.slice(label_tensor, [0, num_classes * FLAGS.update_batch_size, 0], [-1, -1, -1])
            input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb}
        else:
            random.seed(6)
            if FLAGS.datasource == 'plainmulti':
                image_tensor, label_tensor = data_generator.make_data_tensor_plainmulti(train=False)
            elif FLAGS.datasource == 'artmulti':
                image_tensor, label_tensor = data_generator.make_data_tensor_artmulti(train=False)
            inputa = tf.slice(image_tensor, [0, 0, 0], [-1, num_classes * FLAGS.update_batch_size, -1])
            inputb = tf.slice(image_tensor, [0, num_classes * FLAGS.update_batch_size, 0], [-1, -1, -1])
            labela = tf.slice(label_tensor, [0, 0, 0], [-1, num_classes * FLAGS.update_batch_size, -1])
            labelb = tf.slice(label_tensor, [0, num_classes * FLAGS.update_batch_size, 0], [-1, -1, -1])
            metaval_input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb}
    else:
        input_tensors = None
        metaval_input_tensors = None

    print('创建EMO模型...')
    model = EMO(sess, dim_input, dim_output, test_num_updates=test_num_updates)

    if FLAGS.train:
        model.construct_model(input_tensors=input_tensors, prefix='metatrain_')
    else:
        model.construct_model(input_tensors=metaval_input_tensors, prefix='metaval_')

    if FLAGS.train or not FLAGS.test_set:
        model.construct_model(input_tensors=metaval_input_tensors, prefix='metaval_')

    exp_string = f'emo_models_{FLAGS.num_ensemble_models}.datasource_{FLAGS.datasource}.cls_{FLAGS.num_classes}'
    exp_string += f'.mbs_{FLAGS.meta_batch_size}.ubs_{FLAGS.update_batch_size}.numstep_{FLAGS.num_updates}'
    exp_string += f'.updatelr_{FLAGS.update_lr}.metalr_{FLAGS.meta_lr}'
    exp_string += f'.uncertainty_weight_{FLAGS.uncertainty_weight}'

    resume_itr = 0
    model_file = None

    tf.global_variables_initializer().run()

    if FLAGS.resume or not FLAGS.train:
        model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/' + exp_string)
        if model_file:
            ind1 = model_file.index('model')
            resume_itr = int(model_file[ind1 + 5:])
            print(f"从迭代 {resume_itr} 恢复模型")

    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=10)

    if model_file:
        saver.restore(sess, model_file)

    if FLAGS.train:
        train(model, saver, sess, exp_string, data_generator, resume_itr)
    else:
        test(model, sess, data_generator)


if __name__ == "__main__":
    main() 