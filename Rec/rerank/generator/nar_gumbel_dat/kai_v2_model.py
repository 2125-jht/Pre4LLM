MODEL_TRANS_ORIGIN='cpp'

import yaml
import logging
import os
import sys

import argparse
import tensorflow as tf

from feature_attr_extract import *
from model import FountainDeepLtrMultiTaskModel

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'predict'], dest='mode', default='train')
parser.add_argument('--dryrun', dest='dryrun', const=True, default=False, nargs='?')
parser.add_argument('--with_kai', action='store_true', default=False)
parser.add_argument('--text', default=False)
parser.add_argument('--tower', choices=None, dest='tower', default='False')
parser.add_argument('--with_kai_v2', action='store_false', default=True)
args = parser.parse_known_args()[0]
is_training = args.mode == "train"
print("args: ", args)
LIST_SIZE = 6
CANDIDATES_SIZE = 60

print_ops = []
if args.with_kai_v2:
    import kai.tensorflow as config
    import tensorflow.compat.v1 as tf
    default_param_attr = config.nn.ParamAttr(initializer=config.nn.UniformInitializer(0.0001),
                                             access_method=config.nn.ProbabilityAccess(100.0),
                                             recycle_method=config.nn.UnseendaysRecycle(delete_after_unseen_days=30, delete_threshold=0.1, allow_dynamic_delete=True))
    config.nn.set_default_param_attr(default_param_attr)

    def filter_mask_wrapper(dataset):
        dataset.add_feature('fountain_fulllink_rerank_index_list', dataset.DENSE, tf.int64, max_length=60)
        dataset.add_feature('fountain_fulllink_rerank_index_weight_list', dataset.DENSE, tf.int64, max_length=60)
        dataset.add_feature('context_info__real_show_index_list', dataset.DENSE, tf.int64, max_length=60)
        dataset.add_feature('context_info__real_show_list', dataset.DENSE, tf.int64, max_length=60)
        dataset.add_feature('fountain_fulllink_rerank_realshow_label_weight_list', dataset.DENSE, tf.int64, max_length=60)
        dataset.add_feature('context_info__first_screen', dataset.DENSE, tf.int64, max_length=60)
        dataset.add_feature('fountain_wtd_label_list', dataset.DENSE, tf.int64, max_length=60)
        dataset.add_feature('fountain_ltr_label_list', dataset.DENSE, tf.int64, max_length=60)
        dataset.add_feature('fountain_ltr_weight_list', dataset.DENSE, tf.int64, max_length=60)
        dataset.add_feature('rerank_list_score_list', dataset.DENSE, tf.float32, max_length=15)
        dataset.add_feature('rerank_list_item_idx_flat_list', dataset.DENSE, tf.int64, max_length=90)
        dataset.add_feature('photo_info__duration_ms', dataset.DENSE, tf.int64, max_length=60)
        dataset.add_feature('context_info__click_list', dataset.DENSE, tf.int64, max_length=60)
        dataset.add_feature('context_info__like_list', dataset.DENSE, tf.int64, max_length=60)
        dataset.add_feature('context_info__follow_list', dataset.DENSE, tf.int64, max_length=60)
        dataset.add_feature('context_info__comment_list', dataset.DENSE, tf.int64, max_length=60)
        dataset.add_feature('context_info__fountain_slide_to_next_list', dataset.DENSE, tf.int64, max_length=60)

        def mask_fn(batch):
            realshow = batch["context_info__real_show_list"]
            realshow = tf.RaggedTensor.from_row_splits(realshow[0], row_splits=realshow[1])
            realshow = realshow.to_tensor()

            realshow_weight = batch["fountain_fulllink_rerank_realshow_label_weight_list"]
            realshow_weight = tf.RaggedTensor.from_row_splits(realshow_weight[0], row_splits=realshow_weight[1])
            realshow_weight = realshow_weight.to_tensor()

            mean_play_time = tf.reduce_sum(realshow_weight, axis=-1) - 60
            realshow = tf.reduce_sum(realshow, axis=-1)

            mask = tf.math.less(realshow, 2)
            return mask
        return mask_fn
    config.declare_sample_filter(filter_mask_wrapper, data_source_name='train')
else:
    import tensorflow as tf
    from mio_tensorflow.config import MioConfig
    if not args.dryrun and not args.with_kai:
        import mio_tensorflow.patch as mio_tensorflow_patch
        mio_tensorflow_patch.apply()
    logging.basicConfig()
    base_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), './base.yaml')
    config = MioConfig.from_base_yaml(base_config,
                                      clear_embeddings=True,
                                      clear_params=True,
                                      dryrun=args.dryrun,
                                      label_with_kv=True,
                                      grad_no_scale=False,
                                      with_kai=args.with_kai,
                                  predict=(args.mode != "train"))

print("common_attr_names: ", [attr.attr_name for attr in all_features if attr.is_common])
print("all_feature_name: ", [attr.attr_name for attr in all_features])

def mark_common_attr():
    common_embeddings = []
    for attr in all_features:
        if attr.is_common:
            common_embeddings.append(attr.attr_name)
    with open('./infer/dnn_model.yaml', "r+") as f:
        yaml_config = yaml.load(f.read(), Loader=yaml.FullLoader)
        print(yaml_config['embedding']['slots_config'][0])
        for idx, slot_config in enumerate(yaml_config['embedding']['slots_config']):
            if slot_config['input_name'] in common_embeddings:
                yaml_config['embedding']['slots_config'][idx]['is_common'] = True
        f.seek(0)
        yaml.dump(yaml_config, f)
        f.truncate()

def get_param_dict():
    feature_emb_dict = {}
    feature_emb_size_dict = {}
    for attr in all_features:
        print("--->>> feature %s start" % attr.attr_name)
        if not is_training:
            if not attr.expand:
                attr.expand = 1
            embed = None
            if attr.is_common:
                embed = config.new_embedding(attr.attr_name, dim=attr.dim, slots=attr.slots, expand=attr.expand, compress_group='USER')
            else:
                embed = config.new_embedding(attr.attr_name, dim=attr.dim, slots=attr.slots, expand=attr.expand)
            feature_emb_dict[attr.attr_name] = embed
            feature_emb_size_dict[attr.attr_name] = 1
        else:
            if attr.attr_name in photo_fea_names:
                if not attr.expand:
                    attr.expand = CANDIDATES_SIZE
                else:
                    attr.expand *= CANDIDATES_SIZE
            else:
                if args.with_kai:
                    if not attr.expand:
                        attr.expand = 1
            print("attr ", attr, "attr.dim ", attr.dim, "attr.slots ", attr.slots, "attr.expand", attr.expand)
            feature_emb_dict[attr.attr_name] = config.new_embedding(attr.attr_name, dim=attr.dim, slots=attr.slots, expand=attr.expand)
        if attr.expand is not None and attr.expand > 1:
            feature_emb_dict[attr.attr_name] = tf.reshape(feature_emb_dict[attr.attr_name], [-1, attr.expand, attr.dim])

        if args.with_kai_v2:
            sparse_feature = config.get_sparse_fea(name=str(attr.slots[0]))
            offset = sparse_feature[1]
            size_var = offset[1:] - offset[0:-1]
            feature_emb_size_dict[attr.attr_name] = size_var
            if attr.slots[0] == 16:
                tt = tf.RaggedTensor.from_row_splits(values=sparse_feature[0], row_splits=sparse_feature[1]).to_tensor()
        elif args.with_kai:
            offset = tf.cast(config.get_signs(attr.slots[0])[1], tf.int32)
            size_var = offset[1:] - offset[0:-1]
            feature_emb_size_dict[attr.attr_name] = size_var
        print("--->>> feature {} = {}, shape={}".format(attr.attr_name, feature_emb_dict[attr.attr_name], feature_emb_dict[attr.attr_name].shape))

    return feature_emb_dict, feature_emb_size_dict


worker_global_step = config.get_step()
ops = [tf.print("====> step", worker_global_step, summarize=-1, output_stream=sys.stdout)]

all_param_dict, feature_emb_size_dict = get_param_dict()

label_value_dict = {}
label_value_dict["fountain_fulllink_rerank_index_list"] = config.get_label("fountain_fulllink_rerank_index_list", dim=CANDIDATES_SIZE)
label_value_dict["fountain_fulllink_rerank_index_weight_list"] = config.get_label("fountain_fulllink_rerank_index_weight_list", dim=CANDIDATES_SIZE)
label_value_dict["fountain_fulllink_rerank_realshow_label_weight_list"] = config.get_label("fountain_fulllink_rerank_realshow_label_weight_list", dim=CANDIDATES_SIZE)
label_value_dict["fountain_wtd_label_list"] = config.get_label("fountain_wtd_label_list", dim=CANDIDATES_SIZE)
label_value_dict["fountain_ltr_label_list"] = config.get_label("fountain_ltr_label_list", dim=CANDIDATES_SIZE)

dense_dim = CANDIDATES_SIZE if is_training else 1
label_value_dict["context_info__real_show_index_list"] = config.get_dense_fea("context_info__real_show_index_list", dim=CANDIDATES_SIZE, dtype=tf.int64)
label_value_dict["context_info__real_show_list"] = config.get_dense_fea("context_info__real_show_list", dim=CANDIDATES_SIZE, dtype=tf.int64)
label_value_dict["context_info__playing_time_list"] = config.get_dense_fea("context_info__playing_time_list", dim=dense_dim, dtype=tf.int64)
label_value_dict["context_info__click_list"] = config.get_dense_fea("context_info__click_list", dim=CANDIDATES_SIZE, dtype=tf.int64)
label_value_dict["context_info__like_list"] = config.get_dense_fea("context_info__like_list", dim=CANDIDATES_SIZE, dtype=tf.int64)
label_value_dict["context_info__follow_list"] = config.get_dense_fea("context_info__follow_list", dim=CANDIDATES_SIZE, dtype=tf.int64)
label_value_dict["context_info__comment_list"] = config.get_dense_fea("context_info__comment_list", dim=CANDIDATES_SIZE, dtype=tf.int64)
label_value_dict["context_info__fountain_slide_to_next_list"] = config.get_dense_fea("context_info__fountain_slide_to_next_list", dim=CANDIDATES_SIZE, dtype=tf.int64)

loss_names = ["ltr", "vtr", "next"]
model_class = FountainDeepLtrMultiTaskModel(all_param_dict, label_value_dict, loss_names, print_ops)
output_dict, output_dict_gen, gen_loss, cl_loss, predict, item_embeddings, gt_label, dat_loss, transition_matrix, token_distribution, valid_links = model_class.model()

def set_zero_topk(pred, indices):
    batch_size, seq_len, vocab_len = tf.shape(pred)[0], pred.shape[1], pred.shape[2]
    index = tf.expand_dims(tf.range(0, batch_size), axis=1) * seq_len * vocab_len + tf.expand_dims(tf.range(0, seq_len), axis=0) * vocab_len
    index = tf.expand_dims(index, axis=2)
    selected_token = tf.expand_dims(tf.expand_dims(indices, axis=1), axis=2)
    selected_token = tf.cast(selected_token, tf.int32)
    selected_token = tf.tile(selected_token, [1, seq_len, 1]) + index
    pred = tf.reshape(pred, (batch_size * seq_len * vocab_len, 1))
    selected_token = tf.reshape(selected_token, (batch_size * seq_len * 1, 1))
    output_tensor = tf.tensor_scatter_nd_update(pred, selected_token, tf.expand_dims(tf.ones(batch_size * seq_len) * float("-inf"), axis=1))
    output_tensor = tf.reshape(output_tensor, [batch_size, seq_len, vocab_len])
    return output_tensor

def greedy_search(prediction):
    generated_tokens = []
    for i in range(LIST_SIZE):
        logits = prediction[:, i, :]
        _, sampled_token = tf.nn.top_k(logits, k=1)
        sampled_token = tf.squeeze(sampled_token, axis=-1)
        prediction = set_zero_topk(prediction, sampled_token)
        generated_tokens.append(sampled_token)
    generated_tokens = tf.stack(generated_tokens, axis=-1)
    return generated_tokens

def inference_lookahead_repeatprevent(transition_matrix, token_distribution, target_length=6, lookahead=False):
    if lookahead:
        token_distribution_ = tf.reduce_max(token_distribution, axis=-1)
        token_distribution_ = tf.expand_dims(token_distribution_, axis=1)
        transition_matrix = transition_matrix + token_distribution_

    batch_size = tf.shape(transition_matrix)[0]
    num_tokens = tf.shape(token_distribution)[-1]

    output_tokens = []
    used_mask = tf.zeros([batch_size, num_tokens], dtype=tf.bool)

    tokens = tf.argmax(token_distribution, axis=-1)
    first_token = tf.cast(tokens[:, 0], dtype=tf.int32)
    output_tokens.append(first_token)
    used_mask = tf.tensor_scatter_nd_update(
        used_mask,
        tf.stack([tf.range(batch_size), first_token], axis=1),
        tf.ones([batch_size], dtype=tf.bool)
    )
    cur_vertex = tf.cast(tf.argmax(transition_matrix[:, 0, :], axis=-1), dtype=tf.int32)
    for i in range(1, target_length):
        indices = tf.stack([tf.range(batch_size), cur_vertex], axis=1)
        current_token_distribution = tf.gather_nd(token_distribution, indices)
        masked_token_distribution = tf.where(
            used_mask,
            tf.fill([batch_size, num_tokens], -float('inf')),
            current_token_distribution
        )
        next_token = tf.cast(tf.argmax(masked_token_distribution, axis=-1), dtype=tf.int32)
        output_tokens.append(tf.cast(next_token, dtype=tf.int32))
        used_mask = tf.tensor_scatter_nd_update(
            used_mask,
            tf.stack([tf.range(batch_size), next_token], axis=1),
            tf.ones([batch_size], dtype=tf.bool)
        )
        cur_vertex = tf.map_fn(
                            lambda x: tf.cast(tf.argmax(transition_matrix[x, :, :], axis=-1), tf.int32)[cur_vertex[x]],
                            tf.range(batch_size),
                            dtype=tf.int32
                        )
    generated_tokens = tf.stack(output_tokens, axis=1)
    return generated_tokens

def inference_beam_search(transition_matrix, token_distribution, target_length=6, beam_width=10):
    batch_size = tf.cast(tf.shape(token_distribution)[0], dtype=tf.int32)
    num_vertices = tf.cast(tf.shape(token_distribution)[1], dtype=tf.int32)
    num_tokens = tf.cast(tf.shape(token_distribution)[2], dtype=tf.int32)

    initial_logits = tf.gather_nd(token_distribution,
                                tf.stack([tf.range(batch_size, dtype=tf.int32),
                                tf.zeros((batch_size,), dtype=tf.int32)], axis=1))
    log_probs, initial_tokens = tf.nn.top_k(initial_logits, k=beam_width)

    beam_tokens = tf.expand_dims(initial_tokens, axis=2)
    beam_scores = log_probs
    beam_vertices = tf.zeros((batch_size, beam_width), dtype=tf.int32)

    used_tokens = tf.zeros((batch_size, beam_width, num_tokens), dtype=tf.bool)
    batch_indices = tf.range(batch_size)[:, tf.newaxis]
    beam_indices = tf.range(beam_width)[tf.newaxis, :]

    indices = tf.stack([
        tf.tile(batch_indices, [1, beam_width]),
        tf.tile(beam_indices, [batch_size, 1]),
        initial_tokens
    ], axis=-1)

    used_tokens = tf.tensor_scatter_nd_update(
        used_tokens,
        tf.reshape(indices, [-1, 3]),
        tf.ones([batch_size * beam_width], dtype=tf.bool)
    )

    for i in range(1, target_length):
        flat_vertices = tf.reshape(beam_vertices, [-1])
        batch_indices = tf.repeat(tf.range(batch_size), beam_width)
        flat_transition_probs = tf.gather_nd(transition_matrix,
                                           tf.stack([batch_indices, flat_vertices], axis=1))
        flat_token_dist = tf.gather_nd(token_distribution,
                                     tf.stack([batch_indices, flat_vertices], axis=1))
        next_vertex_scores = tf.reshape(flat_transition_probs, [batch_size, beam_width, num_vertices])
        next_token_scores = tf.reshape(flat_token_dist, [batch_size, beam_width, num_tokens])
        next_token_scores = tf.where(
            used_tokens,
            tf.fill(tf.shape(next_token_scores), float('-inf')),
            next_token_scores
        )
        combined_scores = tf.expand_dims(beam_scores, axis=2) + next_token_scores
        flat_scores = tf.reshape(combined_scores, [batch_size, -1])
        top_scores, top_indices = tf.nn.top_k(flat_scores, k=beam_width)
        beam_indices = top_indices // num_tokens
        token_indices = top_indices % num_tokens
        new_tokens = tf.gather(beam_tokens, beam_indices, batch_dims=1)
        new_tokens = tf.concat([new_tokens, tf.expand_dims(token_indices, axis=2)], axis=2)
        next_vertices = tf.argmax(next_vertex_scores, axis=-1)
        next_vertices = tf.gather(next_vertices, beam_indices, batch_dims=1)
        beam_tokens = tf.cast(new_tokens, dtype=tf.int32)
        beam_scores = top_scores
        beam_vertices = tf.cast(next_vertices, dtype=tf.int32)
        used_tokens = tf.gather(used_tokens, beam_indices, batch_dims=1)
        batch_indices = tf.range(batch_size)
        indices = tf.stack([
            tf.repeat(batch_indices, beam_width),
            tf.tile(tf.range(beam_width), [batch_size]),
            tf.reshape(token_indices, [-1])
        ], axis=1)
        used_tokens = tf.tensor_scatter_nd_update(
            used_tokens,
            indices,
            tf.ones([batch_size * beam_width], dtype=tf.bool)
        )

    return beam_tokens


if is_training:
    print(f"====> train, gen...")
    list_dim = 60
    show_label = label_value_dict["context_info__real_show_list"]
    show_label = tf.reshape(show_label, [-1, list_dim])
    show_label = tf.cast(show_label, dtype=tf.float32)
    show_label = show_label[:, :6]
    show_weight = label_value_dict["fountain_fulllink_rerank_realshow_label_weight_list"]
    show_weight = tf.reshape(show_weight, [-1, list_dim])
    show_weight = tf.cast(show_weight, dtype=tf.float32)
    show_weight = show_weight[:, :6]

    wtd_label = label_value_dict["fountain_wtd_label_list"]
    wtd_label = tf.reshape(wtd_label, [-1, list_dim])
    wtd_label = wtd_label[:, :6]
    ltr_label = label_value_dict["fountain_ltr_label_list"]
    ltr_label = tf.reshape(ltr_label, [-1, list_dim])
    ltr_label = ltr_label[:, :6]
    like_label = label_value_dict["context_info__like_list"]
    like_label = tf.reshape(like_label, [-1, list_dim])
    like_label = tf.cast(like_label, dtype=tf.float32)
    like_label = like_label[:, :6]
    follow_label = label_value_dict["context_info__follow_list"]
    follow_label = tf.reshape(follow_label, [-1, list_dim])
    follow_label = tf.cast(follow_label, dtype=tf.float32)
    follow_label = follow_label[:, :6]
    comment_label = label_value_dict["context_info__comment_list"]
    comment_label = tf.reshape(comment_label, [-1, list_dim])
    comment_label = tf.cast(comment_label, dtype=tf.float32)
    comment_label = comment_label[:, :6]
    next_label = label_value_dict["context_info__fountain_slide_to_next_list"]
    next_label = tf.reshape(next_label, [-1, list_dim])
    next_label = tf.cast(next_label, dtype=tf.float32)
    next_label = next_label[:, :6]

    print_ops.append(tf.print("valid_links:", valid_links[0], summarize=5, output_stream=sys.stdout))
    print_ops.append(tf.print("transition_matrix:", transition_matrix[0], summarize=5, output_stream=sys.stdout))
    greedy_indices = inference_lookahead_repeatprevent(valid_links, token_distribution)
    beam_indices = inference_beam_search(valid_links, token_distribution, target_length=6, beam_width=10)
    print_ops.append(tf.print("greedy_indices:", greedy_indices[:5, :6], summarize=5, output_stream=sys.stdout))
    print_ops.append(tf.print("best beam_indices:", beam_indices[:5, 0, :6], summarize=10, output_stream=sys.stdout))

    gen_model_label = label_value_dict['context_info__real_show_index_list']
    gen_model_label = tf.reshape(gen_model_label, [-1, list_dim])
    gen_model_label = tf.cast(gen_model_label, dtype=tf.int32)
    distill_label = tf.cast(tf.math.greater(gen_model_label, 0), tf.float32)
    greedy_hit = tf.batch_gather(distill_label, greedy_indices)
    beam_hit_0th = tf.batch_gather(distill_label, beam_indices[:, 0, :])
    recall_6_th_greedy = tf.reduce_sum(greedy_hit[:, :6], -1, keep_dims=True) / (tf.reduce_sum(distill_label, -1, keep_dims=True) + 1e-9)
    recall_6_th_beam_0 = tf.reduce_sum(beam_hit_0th[:, :6], -1, keep_dims=True) / (tf.reduce_sum(distill_label, -1, keep_dims=True) + 1e-9)

    tf.summary.scalar('recall 6 greedy', tf.reduce_mean(recall_6_th_greedy))
    tf.summary.scalar('recall 6 beam 0', tf.reduce_mean(recall_6_th_beam_0))

    with tf.control_dependencies(print_ops):
        targets = []
        sum_loss_eval = 0.0
        for loss_name in output_dict:
            output = output_dict[loss_name]
            output = tf.reshape(output, [-1, 6])
            if loss_name == "vtr":
                weight_with_mask = show_label
                loss = tf.losses.huber_loss(labels=wtd_label, predictions=output, weights=weight_with_mask,
                                        reduction=tf.losses.Reduction.MEAN,
                                        delta=0.05)
            elif loss_name == "ltr":
                weight_with_mask = show_label
                loss = tf.losses.log_loss(labels=ltr_label, predictions=output, weights=weight_with_mask,
                                        reduction=tf.losses.Reduction.MEAN)
            elif loss_name == "next":
                weight_with_mask = show_label
                loss = tf.losses.log_loss(labels=next_label, predictions=output, weights=weight_with_mask,
                                        reduction=tf.losses.Reduction.MEAN)
            else:
                weight_with_mask = show_label
                loss = tf.losses.log_loss(labels=ltr_label, predictions=output, weights=weight_with_mask,
                                          reduction=tf.losses.Reduction.MEAN)
            sum_loss_eval += loss
            tf.summary.scalar('eval_loss_task_' + loss_name, loss)
        loss_eval = sum_loss_eval / len(loss_names)

        sum_loss_gen_eval = 0.0
        for loss_name in output_dict_gen:
            output = output_dict_gen[loss_name]
            output = tf.reshape(output, [-1, 6])
            if loss_name == "vtr":
                weight_with_mask = show_label
                loss = tf.losses.huber_loss(labels=wtd_label, predictions=output, weights=weight_with_mask,
                                        reduction=tf.losses.Reduction.MEAN,
                                        delta=0.05)
            elif loss_name == "ltr":
                weight_with_mask = show_label
                loss = tf.losses.log_loss(labels=ltr_label, predictions=output, weights=weight_with_mask,
                                        reduction=tf.losses.Reduction.MEAN)
            elif loss_name == "next":
                weight_with_mask = show_label
                loss = tf.losses.log_loss(labels=next_label, predictions=output, weights=weight_with_mask,
                                        reduction=tf.losses.Reduction.MEAN)
            else:
                weight_with_mask = show_label
                loss = tf.losses.log_loss(labels=ltr_label, predictions=output, weights=weight_with_mask,
                                          reduction=tf.losses.Reduction.MEAN)
            sum_loss_gen_eval += loss
            tf.summary.scalar('gen_eval_loss_task_' + loss_name, loss)
        sum_loss_gen_eval = sum_loss_gen_eval / len(loss_names)
        tf.summary.scalar('gen-eval_loss_all_task', sum_loss_gen_eval)

        alpha = 0.5
        generator_loss_sum = sum_loss_gen_eval + cl_loss + alpha * dat_loss
        tf.summary.scalar('generator_cl_loss', cl_loss)
        tf.summary.scalar('generator_gen_loss', dat_loss)
        tf.summary.scalar('generator_total_loss', generator_loss_sum)

        loss_sum = sum_loss_eval
        tf.summary.scalar('evaulator_total_loss', loss_sum)

        item_score = tf.reduce_sum(predict, axis=1)
        predict = tf.identity(predict)
        _, ranked_item_index = tf.math.top_k(item_score, 60, sorted=True)
        item_hit = tf.batch_gather(distill_label, ranked_item_index)
        recall_6_th = tf.reduce_sum(item_hit[:, :6], -1, keep_dims=True) / (tf.reduce_sum(distill_label, -1, keep_dims=True) + 1e-9)
        recall_10_th = tf.reduce_sum(item_hit[:, :10], -1, keep_dims=True) / (tf.reduce_sum(distill_label, -1, keep_dims=True) + 1e-9)
        tf.summary.scalar('recall 6 v1', tf.reduce_mean(recall_6_th))
        tf.summary.scalar('recall 10 v1', tf.reduce_mean(recall_10_th))
        predict = tf.identity(predict)
        pos_output = tf.batch_gather(predict, tf.expand_dims(gt_label, axis=-1))
        pos_output = tf.identity(pos_output)
        pos_output_shape = tf.expand_dims(tf.reduce_sum(pos_output, axis=-1), axis=-1)
        one = tf.ones_like(pos_output_shape)
        zero = tf.zeros_like(pos_output_shape)
        targets.append(('recall_6th', pos_output_shape, zero, one, 'linear_regression'))

    if args.with_kai_v2:
        sparse_optimizer = config.optimizer.Adam(0.001)
        dense_optimizer = config.optimizer.Adam(0.00005)
        dense_optimizer_gen = config.optimizer.Adam(0.00005)
        total_sparse_var = config.get_sparse_trainable_variables()
        total_dense_var = config.get_dense_trainable_variables()
        dense_gen_var_list = []
        dense_eval_var_list = []
        for var in total_dense_var:
            if "generator" in var.name:
                dense_gen_var_list.append(var)
            else:
                dense_eval_var_list.append(var)
        sparse_optimizer.minimize(generator_loss_sum + loss_sum, var_list=total_sparse_var)
        dense_optimizer.minimize(loss_sum, var_list=dense_eval_var_list)
        dense_optimizer_gen.minimize(generator_loss_sum, var_list=dense_gen_var_list)
        opts = [sparse_optimizer, dense_optimizer, dense_optimizer_gen]
    else:
        optimizer = tf.train.GradientDescentOptimizer(1, name="opt")
        grad_var = optimizer.compute_gradients(loss_sum)
        opt = optimizer.apply_gradients(grad_var)
        opts = [opt]

    if args.dryrun:
        pass
    elif args.with_kai:
        print(f"====> train, with kai")
        config.dump_kai_training_config('./training/conf', targets, loss=loss_sum, text=args.text, init_params_in_tf=True)
    elif args.with_kai_v2:
        config.build_model(optimizer=opts, metrics=targets)
    else:
        config.dump_training_config('./training/conf', targets, opts=opts, text=args.text)
else:
    print("predict shape", predict.shape)
    predict = tf.reduce_sum(predict, axis=-1)
    print("predict shape", predict.shape)
    predict = tf.reshape(predict, [-1, 1])

    targets = []
    pred_output = tf.identity(predict)
    targets.append(("rerank_gen", pred_output))
    q_names, preds = zip(*targets)
    config.dump_predict_config(
        "./infer/",
        targets,
        input_type=3,
        extra_preds=q_names,
    )
    print("====> q_name: ", q_names)
    mark_common_attr()

print(f"====> is_training: {is_training}, tower: {args.tower}, dryrun: {args.dryrun}")
