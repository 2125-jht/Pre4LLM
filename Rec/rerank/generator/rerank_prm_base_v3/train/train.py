# -*- coding: UTF8 -*-
import sys

sys.path.append("../bin/")
import time
import tensorflow as tf
import numpy
import threading

import math
import os
from multiprocessing import Process

tf.flags.DEFINE_string("flag_file", "server_static.flags", "flag file")
tf.flags.DEFINE_string("kuiba_op_library_file", "libkuiba_op.so", "kuiba op library path")
tf.flags.DEFINE_integer("worker_index", -1, "worker index")
tf.flags.DEFINE_string("mode", "train", "running mode")
tf.flags.DEFINE_string("predict_graph_dir", "../predict_graph", "predict graph dir")
tf.flags.DEFINE_string("predict_graph_version", "", "predict graph version")

import kuiba_op_v2, kuiba_pybind, kuiba_utils
import worker_cpu_affinity

worker_num = 3
core_num_per_worker = 7
worker_cpu_affinity_result = worker_cpu_affinity.get_worker_cpu_affinity(worker_num, core_num_per_worker)

from manager.model.model_base import ModelOutputType, ModelOut

FLAGS = tf.flags.FLAGS
numpy.set_printoptions(edgeitems=10)

global_step = 0
global_step_lock = threading.Lock()

################### 需要配置具体实验的模型 ############################
from manager.model.model_hot_rerank import ListModel
from manager.loss.loss_function_hot_rerank import hotReRankLossFuction
loss_function_conf = hotReRankLossFuction
model = ListModel(loss_function_conf)
predict_loss_name = ["scores_n_slide", "scores_n"]

is_open_summary = False
is_filter_invalid_gradient = True # 过滤掉不合法的 gradient， 模型实现用stop_gradient可能会出现的问题，正常不应该有不合法的gradient

##################################################################

def _log(prefix, func):
    print("debug in follow:{}:{}".format(prefix, str(func)))


loss_name_list = model.get_loss_name_list()


def main(_):
    cmd_mode = FLAGS.mode
    FLAGS.mode = kuiba_utils.get_predict_mode()
    predict_process = Process(target=work_process, args=(0,))
    predict_process.start()
    predict_process.join()

    FLAGS.mode = cmd_mode
    worker_map = {}
    for worker_index, cpus in worker_cpu_affinity_result.items():
        worker = Process(target=work_process, args=(worker_index,))
        worker_map[worker_index] = worker
        worker.start()
        pid = worker.pid
        os.system('taskset -pc ' + cpus + ' ' + str(pid))
    while len(worker_map) > 0:
        for worker_index in worker_map.keys():
            worker = worker_map[worker_index]
            if not worker.is_alive():
                print("process is not alive, restart, worker_index:", worker_index)
                worker = Process(target=work_process, args=(worker_index,))
                worker_map[worker_index] = worker
                worker.start()
        time.sleep(0.5)
    print("main process finish!")


def work_process(worker_index):
    worker_name = "tf_worker_" + str(worker_index)
    print("{} started".format(worker_name))

    kuiba_pybind.init(worker_name, FLAGS.flag_file, "")

    dryrun_inputs = map(lambda loss_name: loss_function_conf.gen_dryrun_input(loss_name), loss_name_list)
    kuiba_utils.set_dryrun_to_get_variables(True)
    # dry run的意义是在本地进程初始化占位符？
    _ = model.model(*dryrun_inputs)
    kuiba_utils.set_dryrun_to_get_variables(False)

    network_var_list = kuiba_utils.get_network_variables()
    network_var_names = [var.name for var in network_var_list]
    for i in range(len(network_var_names)):
        tf.summary.histogram("{}".format(network_var_names[i]), network_var_list[i])
        print(network_var_names[i])
    sys.stdout.flush()

    x = tf.placeholder(tf.int64, shape=(), name="x")
    batch_id_tensor = kuiba_op_v2.start_batch_op(x)

    all_parameters_tensor_dict = []
    all_parameters_tensor_list = []
    all_labels_tensor = []
    all_weights = []
    all_dependencies_list = []
    all_is_hasweight = []
    merge_summary = None

    for loss_name in loss_name_list:
        (parameters_tensor_dict, labels_tensor, _, weight) = kuiba_op_v2.pull_sparse_op(batch_id_tensor, loss_name,
                                                                                        with_sample_weights=True)
        has_weight = loss_name in loss_function_conf.get_hasweight_loss_list()
        _log("loss_function_conf.get_hasweight_loss_list()_{}".format(loss_name),
             loss_function_conf.get_hasweight_loss_list())
        all_parameters_tensor_dict.append(parameters_tensor_dict)
        all_labels_tensor.append(labels_tensor)
        all_parameters_tensor_list.append(list(parameters_tensor_dict.values()))
        all_weights.append(weight)
        all_is_hasweight.append(has_weight)

        all_dependencies_list.extend(list(parameters_tensor_dict.values()))
        all_dependencies_list.append(labels_tensor)
        if has_weight:
            all_dependencies_list.append(weight)

    _log("all_is_hasweight", all_is_hasweight)

    pull_network_tensor = kuiba_op_v2.pull_network_op(batch_id_tensor, network_var_list)

    with tf.control_dependencies(
            [pull_network_tensor, *all_dependencies_list]):

        # predict
        all_outputs = model.model(*(all_parameters_tensor_dict)) 
        if len(model.loss_name_list) == 1:
            all_outputs = [all_outputs]
        else:
            all_outputs = list(all_outputs)

        # 如果不是model out 就改写成modelout的格式，兼容之前的模型, 之前输出的都是predictions
        if not isinstance(all_outputs[0], ModelOut):
            all_outputs = list(map(lambda output: ModelOut(output), all_outputs))

        assert len(all_outputs) == len(all_is_hasweight) == len(all_weights) == len(all_labels_tensor)

        all_loss_list = []
        all_push_auc_tensor = []
        all_push_mertic = []
        all_dependencies_list_in_train = []
        all_dependencies_list_in_eval = []

        for index in range(len(loss_name_list)):
            loss_name = loss_name_list[index]
            labels_tensor = all_labels_tensor[index]
            _log("labels_tensor_{}".format(loss_name), labels_tensor)

            model_output = all_outputs[index]
            predictions = model_output.predictions
            _log("output_{}".format(loss_name), predictions)

            has_weight = all_is_hasweight[index]
            loss_tensor = None

            if has_weight:
                weight = all_weights[index]
                _log("has_weight_{}".format(loss_name), weight)
            else:
                _log("no_weight_{}".format(loss_name), loss_name)
                # 默认的lable weight 1.0
                weight = 1.0
                _log("no_weight_{}".format(loss_name), loss_tensor)

            if model_output.model_output_type.value == ModelOutputType.predictions.value:
                loss_tensor = tf.losses.log_loss(labels=labels_tensor, predictions=predictions, weights=weight,
                                          reduction=tf.losses.Reduction.SUM)
            elif model_output.model_output_type.value == ModelOutputType.logits.value:
                logits = model_output.logits
                loss_tensor = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels_tensor, logits=logits, weights=weight,
                                                       reduction=tf.losses.Reduction.SUM)
            elif model_output.model_output_type.value == ModelOutputType.loss.value:
                # model out已经输出loss，则不考虑加权问题
                loss_tensor = model_output.get_loss(weight, labels_tensor)
            else:
                assert loss_tensor is not None, "model_output.model_output_type.value={}".format(
                    model_output.model_output_type.value)

            all_loss_list.append(loss_tensor)

            if kuiba_utils.train_mode():
                push_auc_tensor = kuiba_op_v2.push_auc_op(batch_id_tensor, predictions, loss_tensor, loss_name)
                all_push_auc_tensor.append(push_auc_tensor)

                if model_output.set_push_metric_tensor(loss_name, batch_id_tensor, labels_tensor, loss_tensor):
                    all_push_mertic.append(model_output.push_metric_tensor)
                    all_dependencies_list_in_train.append(model_output.push_metric_tensor)

                if model_output.set_push_tensor_op(loss_name,batch_id_tensor):
                    all_push_mertic.append(model_output.push_tensor_op_list)
                    all_dependencies_list_in_train.append(model_output.push_tensor_op_list)

                _log("loss {} in train_mode is begin".format(loss_name), loss_name)
                parameters_tensor_list = all_parameters_tensor_list[index]
                parameters_name_list = all_parameters_tensor_dict[index].keys()

                _log("loss_tensor " + str(loss_name), loss_tensor)
                _log("parameters_tensor_list " + str(loss_name), parameters_tensor_list)
                _log("network_var_list " + str(loss_name), network_var_list)
                parameters_grad_tensor, single_network_var_list, network_grad_tensor = kuiba_utils.compute_grad(
                    loss_tensor, parameters_tensor_list, network_var_list)

                _log("parameters_grad_tensor " + str(loss_name), parameters_grad_tensor)
                _log("single_network_var_list " + str(loss_name), single_network_var_list)
                _log("network_grad_tensor " + str(loss_name), network_grad_tensor)

                # nobp情形下过滤掉None
                if is_filter_invalid_gradient:
                    assert len(parameters_grad_tensor) == len(
                        parameters_name_list), 'grad_list_size not equal name_list_size'
                    parameters_grad_tensor_dict = dict(zip(parameters_name_list, parameters_grad_tensor))
                    filted_parameters_grad_tensor_dict = {name: parameters_grad_tensor_dict[name] for name in
                                                          parameters_grad_tensor_dict
                                                          if parameters_grad_tensor_dict[name] is not None}
                    parameters_name_list = list(filted_parameters_grad_tensor_dict.keys())
                    parameters_grad_tensor = list(filted_parameters_grad_tensor_dict.values())
                    assert len(parameters_grad_tensor) > 0, 'must have gradient to push '
                    push_sparse_tensor = kuiba_op_v2.push_partial_sparse_op(batch_id_tensor, parameters_grad_tensor,
                                                                            loss_name, parameters_name_list)
                else:
                    for i in range(0,len(list(parameters_name_list))):
                        parameters_name = list(parameters_name_list)[i]
                        # if isinstance(parameters_grad_tensor[i], tf.Tensor) and parameters_grad_tensor[i].dtype.is_floating:
                        #     continue
                        print("ModelDebugGradient:parameters_name={},gradient={}".format(parameters_name,str(parameters_grad_tensor[i])))
                    print("parameters_name_list={}".format(str(list(parameters_name_list))))
                    push_sparse_tensor = kuiba_op_v2.push_sparse_op(batch_id_tensor, parameters_grad_tensor, loss_name)
                push_network_tensor = kuiba_op_v2.push_network_op(batch_id_tensor, network_grad_tensor,
                                                                  loss_name,
                                                                  [var.name for var in single_network_var_list])
                push_auc_tensor = all_push_auc_tensor[index]
                all_dependencies_list_in_train.append(push_sparse_tensor)
                all_dependencies_list_in_train.append(push_network_tensor)
                all_dependencies_list_in_train.append(push_auc_tensor)
                _log("loss {} in train_mode is done".format(loss_name), loss_name)
            elif kuiba_utils.evaluate_mode():
                all_dependencies_list_in_eval.extend(all_push_auc_tensor)

        if kuiba_utils.train_mode():
            with tf.control_dependencies(
                    [*all_dependencies_list_in_train]):
                finish_batch_tensor = kuiba_op_v2.finish_batch_op(batch_id_tensor)
                merge_summary = model.merge_all()
        elif kuiba_utils.evaluate_mode():
            with tf.control_dependencies(
                    [*all_dependencies_list_in_eval]):
                finish_batch_tensor = kuiba_op_v2.finish_batch_op(batch_id_tensor)
                merge_summary = None



    config = tf.ConfigProto(
        device_count={"CPU": 1, "GPU": 0},
        allow_soft_placement=True,
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=10
    )
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        if kuiba_utils.predict_mode():
            if worker_index == 0:
                tf.train.write_graph(sess.graph_def, FLAGS.predict_graph_dir,
                                     'predict_graph.binary.pb.{}'.format(FLAGS.predict_graph_version), as_text=False)
                tf.train.write_graph(sess.graph_def, FLAGS.predict_graph_dir,
                                     'predict_graph.text.pb.{}'.format(FLAGS.predict_graph_version), as_text=True)
            kuiba_utils.verify_network_variables(network_var_list)
            sys.exit()

        with open('{}/predict_graph.text.pb.{}'.format(FLAGS.predict_graph_dir, FLAGS.predict_graph_version), 'r') as f:
            predict_graph_text = f.read()
            kuiba_utils.verify_predict_graph(predict_graph_text, *predict_loss_name)
        with open('{}/predict_graph.binary.pb.{}'.format(FLAGS.predict_graph_dir, FLAGS.predict_graph_version),
                  'rb') as f:
            predict_graph_binary = f.read()

        network_init_weights = [tf.make_tensor_proto(var.eval(), dtype=tf.float32) for var in network_var_list]
        init_worker_tensor = kuiba_op_v2.init_worker_op(worker_name, FLAGS.flag_file, network_var_names,
                                                        network_init_weights, predict_graph_binary)
        sess.run(init_worker_tensor)

        kuiba_utils.verify_network_variables(network_var_list)

        print("------------original network------------")
        for var in network_var_list:
            print(var.name + ": " + str(var.shape))
            print(str(var.eval()))
            print("")

        summary_writer = None
        if kuiba_utils.train_mode() and worker_index == 0:
            summary_writer = tf.summary.FileWriter('../summary', sess.graph)
            # summary_writer = None
        if kuiba_utils.evaluate_mode():
            summary_writer = None

        threads = []
        for i in range(10):
            thread = threading.Thread(target=train_thread, name="train-thread{}".format(i), args=(
                sess, finish_batch_tensor, x, merge_summary, summary_writer, all_loss_list), daemon=True)
            threads.append(thread)

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()


def train_thread(sess, finish_batch_tensor, x,
                 merge_summary, summary_writer, all_loss_list):
    global global_step_lock
    global global_step
    cur_step = 0
    while True:
        with global_step_lock:
            global_step += 1
            cur_step = global_step
        start_time = time.time()
        if is_open_summary and summary_writer is not None:
            assert merge_summary is not None
            res_list = sess.run(
                all_loss_list + [finish_batch_tensor, merge_summary],
                feed_dict={x: 1})
            summary = res_list[-1]
            losses_list = res_list[0:-2]
            summary_writer.add_summary(summary, cur_step)
            summary_writer.flush()
        else:
            res_list = sess.run(
                all_loss_list + [finish_batch_tensor],
                feed_dict={x: 1})
            losses_list = res_list[0:-1]
        duration = time.time() - start_time
        print(
            "step: {}, loss: {}, duration: {}".format(
                cur_step, "\t".join(list(map(lambda x: str(x), losses_list))), duration))


if __name__ == "__main__":
    tf.app.run()
