#!/usr/bin/env python3
# coding=utf-8


import os

import argparse
from dragonfly.common_leaf_dsl import LeafFlow, OfflineRunner
from dragonfly.ext.gsu.gsu_api_mixin import GsuApiMixin
from dragonfly.ext.kuiba.kuiba_api_mixin import KuibaApiMixin
from dragonfly.ext.mio.mio_api_mixin import MioApiMixin
from dragonfly.ext.offline.offline_api_mixin import OfflineApiMixin

from input import user_config, item_config, extra_param_config

current_dir = os.path.dirname(__file__)

parser = argparse.ArgumentParser()
parser.add_argument("--run", dest="run", default=False, action="store_true")
parser.add_argument('--eval', dest="eval", default=False, action='store_true')
args = parser.parse_args()

user_config.update(item_config)
param_config = user_config
print(f"====> param_config: {param_config}")

gen_feas = [
    "real_show_index",
    "photo_id",
    "author_id",
    "author_gender",
    "author_fans_count",
    "is_click",
    "is_like",
    "is_follow",
    "is_forward",
    "is_comment",
    "is_collect",
    "is_hate",
    "is_profile_enter",
    "play_time_ms",
    "comment_stay_time",
    "emp_ctr",
    "emp_ltr",
    "emp_wtr",
    "emp_ftr",
    "emp_lvtr",
    "upload_type",
    "exp_tag",
    "city_level",
    "duration_ms",
    "author_dnn_cluster_id",
    "photo_new_music_fingerprint_id",
    "audit_hot_high_tag_level",
    "hetu_cluster_id",
    "hetu_level_one_tag",
    "hetu_level_two_tag",
    "hetu_level_three_tag",
    "pctr",
    "pltr",
    "pwtr",
    "pftr",
    "plvtr",
    "pptr",
    "psvtr",
    "pwtd_score",
    "pcltr",
    "pcmef",
    "pcmtr",
    "mc_htr",
    "mc_pcestr",
    "mc_pcltr",
    "mc_pcmtr",
    "mc_pctr",
    "mc_pepstr",
    "mc_pltr",
    "mc_plvtr",
    "mc_psvtr",
    "mc_pwtd",
    "mc_pwtr",
    "pctr_index",
    "pltr_index",
    "pftr_index",
    "pwtr_index",
    "pvtr_index",
    "plvtr_index"
]

# global values
MODEL_KEY = "fountain_revisit_model_exp"
SAMPLE_HDFS_PATH = "viewfs://hadoop-lt-cluster/home/reco_analysis/dw/reco_analysis.db/explore_reco_log_sample_rel2_server/p_date=%Y%m%d/*#no_next=ignore&no_current=wait&step=86400"
# SAMPLE_HDFS_PATH = "viewfs://hadoop-lt-cluster/home/reco_analysis/dw/reco_analysis.db/explore_reco_log_sample_rel/p_date=20240903/*#no_next=wait"
if args.run:
    SAMPLE_HDFS_PATH = "viewfs://hadoop-lt-cluster/home/reco_analysis/dw/reco_analysis.db/explore_reco_log_sample_rel/p_date=20240903"
# SAMPLE_HDFS_PATH = "viewfs://hadoop-lt-cluster/home/reco_analysis/dw/reco_analysis.db/fountain_user_author_ltv_daily_data/p_date=2024022[0-9]"
# SAMPLE_HDFS_PATH = "viewfs://hadoop-lt-cluster/home/reco_analysis/dw/reco_analysis.db/fountain_user_author_ltv_daily_new/p_date=20240222/hjz_part"
# SAMPLE_HDFS_PATH = "viewfs://hadoop-lt-cluster/home/reco_analysis/dw/reco_analysis.db/fountain_user_author_ltv_daily_new/p_date=20240222/hjz_part_clean"
# SAMPLE_HDFS_PATH = "viewfs://hadoop-lt-cluster/home/reco_analysis/dw/reco_analysis.db/fountain_user_author_ltv_daily_new/p_date=20240222/part-00614-432c5736-12b2-4110-a4e7-79df2922d836-c000"

SAMPLE_COMMON_ATTRS = [
    {"column_index": 0, "column_name": "user_id", "type": "int"},
    {"column_index": 1, "column_name": "device_id", "type": "string"},
    {"column_index": 2, "column_name": "session_id", "type": "int"},
    {"column_index": 3, "column_name": "llsid", "type": "string"},
    {"column_index": 4, "column_name": "user_gender", "type": "int"},
    {"column_index": 5, "column_name": "user_age_segment", "type": "string"},
    {"column_index": 6, "column_name": "time_ms", "type": "int"},
]
SAMPLE_ITEM_ATTRS = [  
    {"column_index": 7, "column_name": "real_show_index_list", "type": "int_list"},
    {"column_index": 8, "column_name": "photo_id_list", "type": "int_list"},
    {"column_index": 9, "column_name": "author_id_list", "type": "int_list"},
    {"column_index": 10, "column_name": "author_gender_list", "type": "int_list"},
    {"column_index": 11, "column_name": "author_fans_count_list", "type": "int_list"},
    {"column_index": 12, "column_name": "is_click_list", "type": "int_list"},
    {"column_index": 13, "column_name": "is_like_list", "type": "int_list"},
    {"column_index": 14, "column_name": "is_follow_list", "type": "int_list"},
    {"column_index": 15, "column_name": "is_forward_list", "type": "int_list"},
    {"column_index": 16, "column_name": "is_comment_list", "type": "int_list"},
    {"column_index": 17, "column_name": "is_collect_list", "type": "int_list"},
    {"column_index": 18, "column_name": "is_hate_list", "type": "int_list"},
    {"column_index": 19, "column_name": "is_profile_enter_list", "type": "int_list"},
    {"column_index": 20, "column_name": "play_time_ms_list", "type": "int_list"},
    {"column_index": 21, "column_name": "comment_stay_time_list", "type": "int_list"},
    {"column_index": 22, "column_name": "emp_ctr_list", "type": "float_list"},
    {"column_index": 23, "column_name": "emp_ltr_list", "type": "float_list"},
    {"column_index": 24, "column_name": "emp_wtr_list", "type": "float_list"},
    {"column_index": 25, "column_name": "emp_ftr_list", "type": "float_list"},
    {"column_index": 26, "column_name": "emp_lvtr_list", "type": "float_list"},
    {"column_index": 27, "column_name": "upload_type_list", "type": "int_list"},
    {"column_index": 28, "column_name": "exp_tag_list", "type": "string_list"},
    {"column_index": 29, "column_name": "city_level_list", "type": "string_list"},
    {"column_index": 30, "column_name": "duration_ms_list", "type": "int_list"},
    {"column_index": 31, "column_name": "author_dnn_cluster_id_list", "type": "int_list"},
    {"column_index": 32, "column_name": "photo_new_music_fingerprint_id_list", "type": "string_list"},
    {"column_index": 33, "column_name": "audit_hot_high_tag_level_list", "type": "int_list"},
    {"column_index": 34, "column_name": "hetu_cluster_id_list", "type": "int_list"},
    {"column_index": 35, "column_name": "hetu_level_one_tag_list", "type": "int_list"},
    {"column_index": 36, "column_name": "hetu_level_two_tag_list", "type": "int_list"},
    {"column_index": 37, "column_name": "hetu_level_three_tag_list", "type": "int_list"},
    {"column_index": 38, "column_name": "pctr_list", "type": "float_list"},
    {"column_index": 39, "column_name": "pltr_list", "type": "float_list"},
    {"column_index": 40, "column_name": "pwtr_list", "type": "float_list"},
    {"column_index": 41, "column_name": "pftr_list", "type": "float_list"},
    {"column_index": 42, "column_name": "plvtr_list", "type": "float_list"},
    {"column_index": 43, "column_name": "pptr_list", "type": "float_list"},
    {"column_index": 44, "column_name": "psvtr_list", "type": "float_list"},
    {"column_index": 45, "column_name": "pwtd_score_list", "type": "float_list"},
    {"column_index": 46, "column_name": "pcltr_list", "type": "float_list"},
    {"column_index": 47, "column_name": "pcmef_list", "type": "float_list"},
    {"column_index": 48, "column_name": "pcmtr_list", "type": "float_list"},
    {"column_index": 49, "column_name": "mc_htr_list", "type": "float_list"},
    {"column_index": 50, "column_name": "mc_pcestr_list", "type": "float_list"},
    {"column_index": 51, "column_name": "mc_pcltr_list", "type": "float_list"},
    {"column_index": 52, "column_name": "mc_pcmtr_list", "type": "float_list"},
    {"column_index": 53, "column_name": "mc_pctr_list", "type": "float_list"},
    {"column_index": 54, "column_name": "mc_pepstr_list", "type": "float_list"},
    {"column_index": 55, "column_name": "mc_pltr_list", "type": "float_list"},
    {"column_index": 56, "column_name": "mc_plvtr_list", "type": "float_list"},
    {"column_index": 57, "column_name": "mc_psvtr_list", "type": "float_list"},
    {"column_index": 58, "column_name": "mc_pwtd_list", "type": "float_list"},
    {"column_index": 59, "column_name": "mc_pwtr_list", "type": "float_list"},
    {"column_index": 60, "column_name": "pctr_index_list", "type": "int_list"},
    {"column_index": 61, "column_name": "pltr_index_list", "type": "int_list"},
    {"column_index": 62, "column_name": "pftr_index_list", "type": "int_list"},
    {"column_index": 63, "column_name": "pwtr_index_list", "type": "int_list"},
    {"column_index": 64, "column_name": "pvtr_index_list", "type": "int_list"},
    {"column_index": 65, "column_name": "plvtr_index_list", "type": "int_list"},
    {"column_index": 66, "column_name": "max_time", "type": "int"},
    {"column_index": 67, "column_name": "min_time", "type": "int"},
    {"column_index": 68, "column_name": "timelist", "type": "string"},
    {"column_index": 69, "column_name": "d_session_label", "type": "int"},

]

class DataReaderFlow(LeafFlow, MioApiMixin, KuibaApiMixin, OfflineApiMixin, GsuApiMixin):
    def print(self):
        print(self.name)

    def clean_all(self, reason, **kwargs):
        return self.limit(0, name="clean_all_for_" + reason, **kwargs)

    def tab_filter(self):
        # 目前只用双列发现页外流样本
        return self.if_("tab ~= 0").clean_all("skip_not_explore_log").end_()

    def process_feat(self):
        return self.enrich_attr_by_lua(
            import_item_attr=[
                "author_fans_count_list",
                "play_time_ms_list",
                "comment_stay_time_list",
            ],
            export_item_attr=[
                "fans_user_num_log_list",
                "play_time_sec_list",
                "comment_stay_time_sec_list",
            ],
            function_for_item="process_feature",
            lua_script_file="util.lua"
        ) \
        .enrich_attr_by_lua(
            function_for_common='transfer_user_age',
            import_common_attr=['user_age_segment'],
            export_common_attr=['user_age_segment'],
            lua_script="""
                function transfer_user_age()
                    local index = 0
                    if user_age_segment == "AGE_0_12" then
                    index = 1
                    elseif user_age_segment == "AGE_12_17" then
                    index = 2
                    elseif user_age_segment == "AGE_18_23" then
                    index = 3
                    elseif user_age_segment == "AGE_24_30" then
                    index = 4
                    elseif user_age_segment == "AGE_31_40" then
                    index = 5
                    elseif user_age_segment == "AGE_41_49" then
                    index = 6
                    elseif user_age_segment == "AGE_50_INF" then
                    index = 7
                    end
                    return index
                end
                """
            ) \
    # .enrich_attr_by_lua(
    #     function_for_item='transfer_author_gender',
    #     import_item_attr=['author_gender_str'],
    #     export_item_attr=['author_gender'],
    #     lua_script="""
    #         function transfer_author_gender()
    #             local index = 0
    #             if author_gender_str == "M" then
    #             index = 1
    #             elseif author_gender_str == "F" then
    #             index = 2
    #             end
    #             return index
    #         end
    #         """
    #     ) \
    # .enrich_attr_by_lua(
    #     import_item_attr=["author_age_segment_str",],
    #     export_item_attr=["author_age_segment"],
    #     function_for_item="transfer_author_age",
    #     lua_script_file="util.lua"
    # ) \

    def get_remote_pid_emb(self):
        return self.pack_item_attr(
            item_source={
                "reco_results": True,
            },
            mappings=[{
                "aggregator": "copy",
                "from_item_attr": "photo_id",
                "to_common_attr": "common_photo_id",
            }]
        ).get_remote_embedding_lite(
            kess_service="grpc_MMUHetuSimPicContentEmbedding",
            shard_num=4,
            id_converter={"type_name": "kuibaEmbeddingIdConverter"},
            input_attr_name="common_photo_id",
            output_attr_name="hetu_sim_emb",
            query_source_type="common_attr",
            size=64,
            client_side_shard=True,
        ).copy_attr(
            attrs=[{
                "from_common": "hetu_sim_emb",
                "to_item": "photo_id_emb"
            }]
        )

    def gen_label(self):
        return self.get_kconf_params(
            kconf_configs=[
            ],
        ).enrich_attr_by_lua(
            import_common_attr=[
            ],
            import_item_attr=[
                "is_click_list",
                "is_like_list",
                "is_follow_list",
                "is_forward_list",
                "is_comment_list",
                "is_collect_list",
                "is_hate_list",
                "play_time_ms_list",
                "duration_ms_list",
                "max_time",
                "min_time",
                "timelist",
                "d_session_label",
                "real_show_index_list"
            ],
            export_item_attr=[
                "point_ltr_label",
                "point_ltr_wt",
                "session_inner_time",
                "session_vv",
                "is_revisit",
                "is_next_label",
                "is_real_show_list",
            ],
            function_for_item="gen_label",
            lua_script_file="util.lua"
        )


# build a pipeline
load_raw_sample = (
    DataReaderFlow(name="load_raw_sample")
    .fetch_message(
        group_id=MODEL_KEY,
        hdfs_path=SAMPLE_HDFS_PATH,
        hdfs_format="raw_text",
        hdfs_read_thread_num=64,
        output_attr="hive_sample_str"
    ) \
    .filter_by_attr(
        attr_name="hive_sample_str",
        remove_if="==",
        compare_to="",
        remove_if_attr_missing=True,
    ) \
    .convert_csv_to_tf_sequence_example(
        from_extra_var="hive_sample_str",
        common_attrs=SAMPLE_COMMON_ATTRS,
        item_attrs=SAMPLE_ITEM_ATTRS,
        column_separator="|",
        item_separator=",",
        list_separator=" ",
        save_result_to="tf_sequence_example"
    ) \
    .retrieve_from_tf_sequence_example(
        from_extra_var="tf_sequence_example",
        user_id_attr="user_id",
        time_ms_attr="time_ms",
    )
    .log_debug_info(
        # common_attrs = [
        #     "hive_sample_str",
        #     "tf_sequence_example",
        #     "time_ms",
        #     "user_age_segment",
        #     "user_gender",
        # ],
        item_attrs = [
            "timelist"
        ],
        for_debug_request_only= False
    ) \
    .process_feat()
    # .get_remote_pid_emb()
    .gen_label()
    .extract_kuiba_parameter(
        config=param_config,
        slots_output="eyeshot_slots",
        parameters_output="eyeshot_signs",
    ) \
    .log_debug_info(
        common_attrs = [
            "user_id",
            "session_id",
            "time_ms",
            "user_age_segment"
        ],
        item_attrs = [
            "emp_ctr_list",
            "emp_wtr_list",
            "mc_pltr_list",
            "pftr_index_list",
            "hetu_level_one_tag_list",
            "hetu_level_two_tag_list",
            "hetu_level_three_tag_list",
            "is_click_list",
            "is_like_list",
            "is_follow_list",
            "is_forward_list",
            "is_comment_list",
            "is_collect_list",
            "is_hate_list",
            "play_time_ms_list",
            "duration_ms_list",
            "max_time",
            "min_time",
            "timelist",
            "d_session_label",
            "author_id_list",
            "author_gender_list",
            "point_ltr_label", "point_ltr_wt","session_inner_time","session_vv","is_revisit","real_show_index_list","is_next_label","is_real_show_list"
        ],
        for_debug_request_only=False
    ) \
    .log_debug_info(
        common_attrs = [
            "hive_sample_str",
            "tf_sequence_example",
            "time_ms",
            "user_age_segment",
            "user_gender",
        ],
       item_attrs=[
            "eyeshot_slots",
            "eyeshot_signs",
            "author_age_segment",
            "author_age_segment_str",
            "author_gender",
            "author_gender_str",
            "hetu_cluster_id",
            "hetu_level_two_tag",
            "author_id",
            
            "ctr_label",
            "ctr_mask",
            "ctr_weight",
            "cvr_label",
            "cvr_mask",
            "cvr_weight",
            "click_label",
            "click_mask",
            "click_weight",
            "action_label",
            "action_mask",
            "action_weight",
        ],
        print_all_item_attrs=True,
        print_all_item_keys=True,
        for_debug_request_only= (False if args.run else True),
        respect_sample_logging= (False if args.run else True),
    ) \
    )

send_to_mio_learner = (
    DataReaderFlow(name="send_to_mio_learner")
    .copy_user_meta_info(save_result_size_to_attr="item_num")
    .if_("item_num > 0")
    .send_to_mio_learner(
        attrs=[] +
              ["point_ltr_label", "point_ltr_wt","session_inner_time","session_vv","is_revisit", "is_real_show_list", "is_next_label"] +
              list(extra_param_config.keys()),
        slots_attrs=["eyeshot_slots"],
        signs_attrs=["eyeshot_signs"],
        lineid_attr="user_id",
        user_hash_attr="device_id",
        label_attr="ltr",
        time_ms_attr="time_ms"
    )
    .end_if_()
)
# 对应一个离线 Pipeline Runner，当前仅支持一个 LeafFlow。
runner = OfflineRunner(MODEL_KEY)
runner.ENABLE_ATTR_CHECK = False
if args.run:
    runner.add_leaf_flows(leaf_flows=[load_raw_sample])
    exe = runner.executor()
    for i in range(50000):
        exe.reset()
        exe.run("load_raw_sample")
elif args.eval:
    runner.add_leaf_flows(leaf_flows=[load_raw_sample, send_to_mio_learner])
    runner.build(output_file=os.path.join(current_dir, "cofea_reader_eval.json"))
else:
    runner.add_leaf_flows(leaf_flows=[load_raw_sample, send_to_mio_learner])
    runner.build(output_file=os.path.join(current_dir, "cofea_reader.json"))


