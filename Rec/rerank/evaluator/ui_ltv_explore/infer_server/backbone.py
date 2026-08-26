import base64
import collections
import json
import os
import sys

import yaml

current_dir = os.path.dirname(__file__)
from dragonfly.common_leaf_dsl import LeafService, LeafFlow
from dragonfly.ext.offline.offline_api_mixin import OfflineApiMixin
from dragonfly.ext.kuiba.kuiba_api_mixin import KuibaApiMixin
from dragonfly.ext.mio.mio_api_mixin import MioApiMixin
from dragonfly.ext.gsu.gsu_api_mixin import GsuApiMixin
from dragonfly.ext.cofea.cofea_api_mixin import CofeaApiMixin
from dragonfly.ext.uni_predict.uni_predict_api_mixin import UniPredictApiMixin
from dragonfly.ext.oversea.oversea_api_mixin import OverseaApiMixin

kess_name = "grpc_RerankGenNextModelLtrInfer"

return_labels = [
    "pltr0", "pltr1", "pltr2", "pltr3", "pltr4", "pltr5","pltr6","pltr7", "pltr8", "pltr9",
    "pinner","pvv","previsit"]

batch_sizes = [600, 500, 200]

# load Resources
ModelConfig = collections.namedtuple(
    'ModelConfig',
    ['graph', 'outputs', 'slots_config', 'param', 'common_parameter_config_rename', 'non_common_parameter_config_rename', 'vec_input']
)
all_attrs = set()
all_features = {}


def load_mio_tf_model(model_dir):
    with open(os.path.join(model_dir, 'dnn_model.yaml')) as f:
        dnn_model = yaml.load(f, Loader=yaml.SafeLoader)

    with open(os.path.join(model_dir, 'graph.pb'), 'rb') as f:
        base64_graph = base64.b64encode(f.read()).decode('ascii')
        graph = 'base64://' + base64_graph

    graph_tensor_mapping = dnn_model['graph_tensor_mapping']
    extra_preds = dnn_model['extra_preds'].split(' ')
    q_names = dnn_model['q_names'].split(' ')
    assert len(extra_preds) == len(q_names)
    outputs = [(extra_pred, graph_tensor_mapping[q_name]) for extra_pred, q_name in zip(extra_preds, q_names)]
    param = [param for param in dnn_model['param'] if param.get('send_to_online', True)]

    slots_config = dnn_model['embedding']['slots_config']
    vec_input = dnn_model['vec_input']
    for c in vec_input:
        print(f"====> name: {c['name']}, tensor_name: {c['name']}, common: {c['common']}, dim: {c['dim']}")

    global all_attrs
    global all_features
    
    print('current_dir', current_dir)
    feature_config = json.load(open(os.path.join(current_dir, "./feature_config.json")))
    parameter_config = feature_config['common_features']
    common_slots = [f['attrs'][0]['key_type'] for f in parameter_config.values()]
    print('====> common slots: ' + ' '.join(map(str, common_slots)))
    parameter_item_config = feature_config['item_features']
    item_slots = [f['attrs'][0]['key_type'] for f in parameter_item_config.values()]
    print('====> item slots: ' + ' '.join(map(str, item_slots)))

    common_parameter_config_rename = dict()
    non_common_parameter_config_rename = dict()

    for name, feature_config in parameter_config.items():
        common_parameter_config_rename[name] = feature_config
        extrators = feature_config['attrs']
        for extrator in extrators:
            if extrator['converter'] == 'combine':
                converter_args = extrator['converter_args']
                attr = [*converter_args['left'].keys(), *converter_args['right'].keys()]
            else:
                attr = extrator['attr']
            all_attrs.update(attr)
    for name, feature_config in parameter_item_config.items():
        non_common_parameter_config_rename[name] = feature_config
        extrators = feature_config['attrs']
        for extrator in extrators:
            if extrator['converter'] == 'combine':
                converter_args = extrator['converter_args']
                attr = [*converter_args['left'].keys(), *converter_args['right'].keys()]
            else:
                attr = extrator['attr']
            all_attrs.update(attr)
    print("====> common_parameter_config_rename", common_parameter_config_rename)
    print("====> non_common_parameter_config_rename", non_common_parameter_config_rename)

    for slot_config in slots_config:
        input_name = slot_config["input_name"]
        all_features[input_name] = 1
        print(f"====> slot_conf: {slot_config}")

    return ModelConfig(graph, outputs, slots_config, param, common_parameter_config_rename, non_common_parameter_config_rename, vec_input)


# load Mixins
class PredictServerFlow(LeafFlow, KuibaApiMixin, MioApiMixin, OfflineApiMixin, GsuApiMixin, CofeaApiMixin, UniPredictApiMixin, OverseaApiMixin):
    def predict_with_mio_model(self, **kwargs):
        model_config = kwargs.pop('model_config')
        embedding_kess_name = kwargs.pop('embedding_kess_name')
        queue_prefix = kwargs.pop('queue_prefix')
        key = kwargs.pop('key', queue_prefix)
        receive_dnn_model_as_macro_block = kwargs.pop('receive_dnn_model_as_macro_block', False)
        extra_inputs = kwargs.pop('extra_inputs', [])
        shards = kwargs.pop('shards', 1)
        rowmajor = kwargs.pop('rowmajor', True)
        extra_signs = kwargs.pop('extra_signs', [])
        extra_slots = kwargs.pop('extra_slots', [])

        slots_config = []
        for c in model_config.slots_config:
            if 'dtype' in c:
                # dtype 默认为 mio_int16
                # 支持 mio_int16, scale_int8, scale_int16, float16, float32 需要根据自己的 embedding server 确定
                c['dtype'] = 'mio_int16'
            slots_config.append(c)
        return (self
            .get_abtest_params(
                biz_name="RECO_RPC",
                ab_params=[
                    # ("xxx_v1", 0, "xxx"),
                ]
            ) \
            .log_debug_info(
                print_all_item_attrs = True,
                print_all_item_keys =True,
                for_debug_request_only = False,
                respect_sample_logging = True,
            ) \
            .copy_item_meta_info(
              save_item_id_to_attr = "item_id",
            ) \
            .copy_user_meta_info(
              save_user_id_to_attr = "user_id",
            ) \
            .get_item_attr_by_distributed_flat_index(
              photo_store_kconf_key = "reco.distributedIndex.explorePhotoInfoCommon",
              use_dynamic_photo_store = True,
              item_id_attr = "item_id",
              attrs = [
                 "photo_id",
                 { "name": "hetu_tag_level_info__hetu_cluster_id", "as": "hetu_cluster_id_index"},
                 { "name": "photo_id", "as": "photo_id_index"},
                 { "name": "author__id", "as": "author_id"},
                 { "name": "author__gender", "as": "author_gender"},
                 { "name": "author_age_info__age_segment", "as": "author_age_segment"},
                 { "name": "author__fans_count", "as": "author_fans_count"},
                 { "name": "hetu_tag_level_info__hetu_level_two", "as": "hetu_level_two_tag_index"},
              ],
            ) \
            # .log_debug_info(
            #     item_attrs=[
            #         "hetu_level_two_tag",
            #         "hetu_level_two_tag_index"
            #     ],
            #     for_debug_request_only = False,
            #     respect_sample_logging = False
            # )
            # .count_reco_result(
            #     save_count_to="hetu_level_two_tag_cnt_null",
            #     select_item = {
            #     "attr_name": "hetu_level_two_tag",
            #     "select_if": "is null",
            #     }
            # )
             .enrich_attr_by_lua(
                function_for_item='transfer_hetu_level_two',
                import_item_attr=['hetu_level_two_tag'],
                export_item_attr=['hetu_level_two_tag_rename'],
                lua_script="""
                    function transfer_hetu_level_two()
                        if hetu_level_two_tag ~= nil then
                            return table.concat(hetu_level_two_tag, "_")
                        else
                            return ""
                        end
                    end
                    """
                ) \
            .enrich_attr_by_lua(
                import_item_attr=[
                    "pctr_list",
                    "pltr_list",
                    "pwtr_list",
                    "plvtr_list",
                    "pcmtr_list",
                    "pcmef_list",
                    "pptr_list",
                    "pctr_index_list",
                    "pltr_index_list",
                    "pwtr_index_list",
                    "pvtr_index_list",
                    "plvtr_index_list",
                    "emp_ctr_list",
                    "emp_ltr_list",
                    "emp_wtr_list",
                    "emp_lvtr_list"
                ],
                export_item_attr=[
                    "pctr_list_dense",
                    "pltr_list_dense",
                    "pwtr_list_dense",
                    "plvtr_list_dense",
                    "pcmtr_list_dense",
                    "pcmef_list_dense",
                    "pptr_list_dense",
                    "pctr_index_list_dense",
                    "pltr_index_list_dense",
                    "pwtr_index_list_dense",
                    "pvtr_index_list_dense",
                    "plvtr_index_list_dense",
                    "emp_ctr_list_dense",
                    "emp_ltr_list_dense",
                    "emp_wtr_list_dense",
                    "emp_lvtr_list_dense"
                ],
                function_for_item="gen_item_attr",
                lua_script_file="util.lua"
            ) \
            .extract_kuiba_parameter(slots_output="slots", parameters_output="parameters", is_common_attr=False, config=model_config.non_common_parameter_config_rename)
            .extract_kuiba_parameter(slots_output="common_slots", parameters_output="common_parameters", is_common_attr=True, config=model_config.common_parameter_config_rename)
            .uni_predict_fused(
                ## 拉取 embedding 配置
                embedding_fetchers=[dict(
                    fetcher_type="BtEmbeddingServerFetcher",
                    kess_service=embedding_kess_name,
                    shards=shards,
                    slots_inputs=["slots"] + extra_slots,
                    parameters_inputs=["parameters"] + extra_signs,
                    common_slots_inputs=["common_slots"],
                    common_parameters_inputs=["common_parameters"],
                    timeout_ms=50,
                    slots_config=slots_config,
                    client_side_shard=True,
                    max_signs_per_request=4096,
                )],
                ## 图相关配置
                graph=model_config.graph,
                queue_prefix=queue_prefix,
                key=key,
                param=model_config.param,
                inputs=[dict(
                    attr_name=c['input_name'],
                    tensor_name=c['input_name'],
                    # common=c.get('common', False),
                    common=(c.get('compress_group', "") == "USER"),
                    dim=len(str(c['slots']).split(' ')) * c['dim'] * c.get('expand', 1) + (1 if c.get('sized', False) else 0),
                    ) for c in model_config.slots_config] +
                    [dict(
                    attr_name=c['name']+'_dense',
                    tensor_name=c['name'],
                    common=c['common'],
                    dim=c['dim'],) for c in model_config.vec_input],
                outputs=[dict(
                    attr_name=attr_name,
                    tensor_name=tensor_name,) for attr_name, tensor_name in model_config.outputs if attr_name in return_labels],
                ## batching 相关配置
                # cpu 预估也支持 batching，但是由于 cpu 服务性能有限，开 batching 效果未必明显
                batching_config=dict(
                    batch_timeout_micros=5000,
                    max_batch_size=max(batch_sizes),
                    max_enqueued_batches=1,
                    batch_task_type="BatchTensorflowTask",
                ),
                ## 模型加载相关的配置
                model_loader_config=dict(
                    type="MioTFExecutedByTensorFlowModelLoader",
                    executor_batchsizes=batch_sizes,
                    implicit_batch=True,  # cpu 预估推荐使用 implicit batch 预估
                    rowmajor=True,
                    receive_dnn_model_as_macro_block=receive_dnn_model_as_macro_block,
                ),

                ## 执行相关配置
                executor_config=dict(
                    intra_op_parallelism_threads_num=32,
                    inter_op_parallelism_threads_num=32,
                ),
            )
            .enrich_attr_by_lua(
                import_item_attr=[
                    "pctr",
                    "pltr",
                    "pwtr",
                    "ctr",
                    "cvr",
                    "click",
                    "action",
                ],
                export_item_attr=[
                    "revisit",
                    "revisit2",
                ],
                function_for_item="gen_return_attr",
                lua_script_file="util.lua"
            ) 
            .log_debug_info(
                common_attrs=[
                    "user_id",
                    "common_slots",
                    "common_parameters",
                ],
                item_attrs=[
                    "slots",
                    "parameters",
                ],
                print_all_item_attrs=True,
                print_all_item_keys=True,
                for_debug_request_only = False,
                respect_sample_logging = False
            )
            .log_debug_info(
                common_attrs=[
                    "user_id",
                    "common_slots",
                    "common_parameters",
                ],
                item_attrs=[
                    "pctr_list",
                    "pltr_list",
                    "pwtr_list",
                    "plvtr_list",
                    "pcmtr_list",
                    "pcmef_list",
                    "pptr_list",
                    "pctr_index_list",
                    "pltr_index_list",
                    "pwtr_index_list",
                    "pvtr_index_list",
                    "plvtr_index_list",
                    "emp_ctr_list",
                    "emp_ltr_list",
                    "emp_wtr_list",
                    "emp_lvtr_list",
                    "pctr_list_dense",
                    "pltr_list_dense",
                    "pwtr_list_dense",
                    "plvtr_list_dense",
                    "pcmtr_list_dense",
                    "pcmef_list_dense",
                    "pptr_list_dense",
                    "pctr_index_list_dense",
                    "pltr_index_list_dense",
                    "pwtr_index_list_dense",
                    "pvtr_index_list_dense",
                    "plvtr_index_list_dense",
                    "emp_ctr_list_dense",
                    "emp_ltr_list_dense",
                    "emp_wtr_list_dense",
                    "emp_lvtr_list_dense"
                ],
                for_debug_request_only = False,
            )
            .perflog_attr_value(
                check_point="input_dense",
                item_attrs=[
                    "pctr_list_dense",
                    "pltr_list_dense",
                    "pwtr_list_dense",
                    "plvtr_list_dense",
                    "pcmtr_list_dense",
                    "pcmef_list_dense",
                    "pptr_list_dense",
                    "pctr_index_list_dense",
                    "pltr_index_list_dense",
                    "pwtr_index_list_dense",
                    "pvtr_index_list_dense",
                    "plvtr_index_list_dense",
                    "emp_ctr_list_dense",
                    "emp_ltr_list_dense",
                    "emp_wtr_list_dense",
                    "emp_lvtr_list_dense"
                ],
             )
            )


all_attrs_list = list(sorted(all_attrs))
service = LeafService(kess_name=kess_name)

service.AUTO_INJECT_ITEM_ATTR = False
service.AUTO_INJECT_SAMPLE_LIST_USER_ATTR = False
service.return_item_attrs(return_labels)
