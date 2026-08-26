from cascading import CommonModule
from cascading.module.fountain_fast_cascading_queues import *
from cascading.module.fountain_cascading_utils import cascade_ltr_common_feature

class FountainFastCascadingScoreModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
    .if_("fountain_cascade_ftr_slide_kai_predict_all == 1 or fountain_casade_is_fast == 1") \
      .if_("cascade_ftr_kai_use_ipw_weight_new == 1") \
        .enrich_attr_by_lua(
          import_common_attr = [
            "enable_opt_cascade_ftr_ipw_bucket",
            "ftr_redis_key_opt_prefix"
          ],
          import_item_attr = [
            "duration_ms",
            "cascade_ftr_kai",
          ],
          export_item_attr = [
            "cascade_ftr_kai_redis_key"
          ],
          function_for_item = "cascade_ftr_redis_key",
          lua_script_file = "./life/cascading/lua/module/fountain_splash_cascading_predict__calc_pxtr.lua",
        ) \
        .get_item_attr_from_redis(
          cluster_name = "recoNewUserPhotos",
          timeout_ms = 50,
          cache_bits = 16,
          redis_key_from="cascade_ftr_kai_redis_key",
          save_value_to="cascade_ftr_kai_ipw_value",
        ) \
        .enrich_attr_by_lua(
          import_common_attr = [
            "cascade_ftr_ipw_debias_v1",
            "cascade_ftr_ipw_debias_v2",
            "cascade_ftr_ipw_debias_v3",
            "cascade_ftr_kai_ipw_value_default",
            "cascade_ftr_ipw_debias_ftr_alpha",
            "cascade_ftr_ipw_debias_ftr_factor",
            "cascade_ftr_ipw_debias_ftr_beta",
            "cascade_ftr_ipw_debias_pct_beta"
          ],
          import_item_attr = [
            "cascade_ftr_kai",
            "cascade_ftr_kai_ipw_value",
          ],
          export_item_attr = [
            "cascade_ipw_opt_ftr"
          ],
          function_for_item = "cascade_ftr_ipw_debias",
          lua_script_file = "./life/cascading/lua/module/fountain_splash_cascading_predict__calc_pxtr.lua",
        ) \
      .end_if_() \
      .if_("fountain_skip_calc_cascade_ftr_duration == 0") \
        .enrich_attr_by_lua( # 结合 duration 的纠偏
          import_common_attr = [
            "cascade_ftr_kai_duration_max",
            "cascade_ftr_kai_duration_weight",
            "cascade_ftr_kai_duration_min",
            "cascade_ftr_kelly_avg_duration",
            "cascade_ftr_kai_enable_transfer_1",
            "cascade_ftr_kai_enable_transfer_2",
            "cascade_ftr_kai_duration_ftr_power",
            "cascade_ftr_kai_duration_ftr_offset",
          ],
          import_item_attr = [
            "duration_ms",
            "cascade_ftr_kai"
          ],
          export_item_attr = [
            "cascade_ftr_kai_duration",
            "cascade_ftr_kai_kelly",
          ],
          function_for_item = "cascade_ftr_duration",
          lua_script_file = "./life/cascading/lua/module/fountain_splash_cascading_predict__calc_pxtr.lua",
        ) \
        .log_debug_info(
          common_attrs = [
            "cascade_ftr_kai_duration_max",
            "cascade_ftr_kai_duration_weight",
            "cascade_ftr_kai_duration_min",
            "cascade_ftr_kelly_avg_duration"
          ],
          item_attrs = [
            "cascade_ftr_kai_duration",
            "cascade_ftr_kai_kelly",
          ],
          item_num_limit = 10,
          for_debug_request_only = True,
        ) \
      .end_if_() \
      .if_("fountain_skip_calc_cascade_ftr_diff_score == 0") \
        .transform_item_attr(  # 判断是否是图片
          mappings = [{
            "check_attr_name": "upload_type",
            "check_attr_type": "int",
            "output_attr_name": "is_picture",
            "output_attr_type": "int",
            "rules": [{
              "check_values": [7, 10, 11, 70],
              "output_value": 1
            }]
          }]) \
        .transform_item_attr(  # 判断是否是图片
          mappings = [{
            "check_attr_name": "duration_ms",
            "check_attr_type": "int",
            "output_attr_name": "is_picture",
            "output_attr_type": "int",
            "rules": [{
              "check_range": {
                "upper_bound": 101, # 不包含
              },
              "output_value": 1
            }]
          }]) \
        .enrich_attr_by_lua( # 计算平均完播率的时长分桶
          import_item_attr = [
            "duration_ms",
            "is_picture"
          ],
          export_item_attr = [
            "duration_cluster_id"
          ],
          function_for_item = "calc_cascade_duration_cluster_id",
          lua_script_file = "./life/cascading/lua/module/fountain_fast_cascading_score__calc_cascade_interest_cluster_id.lua",
        ) \
        .get_kconf_params(
          kconf_configs = [{
            "kconf_key": "reco.hot.fountainDurationCluster2AvgFtrJson",
            "json_path": "{{duration_cluster_id}}",
            "default_value": 0.0,
            "export_item_attr": "duration_cluster_default_ftr"
          }]
        ) \
        .enrich_attr_by_lua(
          import_common_attr = [
            "fountain_cascade_ftr_kai_max_value",
            "fountain_avg_ftr_min_click_count",
            "fountain_cascade_ftr_diff_score_bias"
          ],
          import_item_attr = [
            "cascade_ftr_kai",
            "duration_ms",
            "is_picture",
            "duration_cluster_default_ftr",
            "explore_stat__view_length_sum",
            "explore_stat__click_count",
          ],
          export_item_attr = [
            "cascade_ftr_emp",
            "cascade_ftr_diff_score"
          ],
          function_for_item = "calc_cascade_ftr_diff_score",
          lua_script_file = "./life/cascading/lua/module/fountain_splash_cascading_predict__calc_pxtr.lua",
        ) \
        .log_debug_info(
          item_attrs = [
            "cascade_ftr_kai",
            "cascade_ftr_emp",
            "cascade_ftr_diff_score",
            "duration_cluster_id",
            "duration_cluster_default_ftr"
          ],
          item_num_limit = 10,
          for_debug_request_only = True,
        ) \
      .end_if_() \
      .if_("fountain_cascade_fintr_transfer_new == 1") \
        .explore_memory_data_enrich(
          data_key = "{{fountain_cascade_pfintr_debias_map}}",
          data_type = "string_double_vector_map",
          save_data_ptr_to_attr = "fintr_debias_map_ptr",
        ) \
        .explore_trans_fintr_enricher(
          enable_transfer_sigmoid = "{{fountain_cascade_enable_transfer_fintr_sigmoid}}",
          get_fintr_quantile_mode = "{{fountain_cascade_get_fintr_quantile_mode}}",
          fintr_debias_map_attr = "fintr_debias_map_ptr",
          fintr_redis_key_prefix = "{{fountain_cascade_fintr_redis_key_prefix}}",
          fintr_short_photo_cluster_dist = "{{fountain_cascade_fintr_photo_cluster_dist}}",
          fintr_long_photo_threshold = "{{fountain_cascade_fintr_long_photo_threshold}}",
          fintr_long_photo_cluster_dist = "{{fountain_cascade_fintr_long_photo_cluster_dist}}",
          max_dura_limit = "{{fountain_cascade_fintr_max_dura_limit}}",
          max_fintr_limit = "{{fountain_cascade_fintr_max_cluster_limit}}",
          fintr_dist_reciprocal = "{{fountain_cascade_fintr_dist_reciprocal}}",
          enable_map_fintr_positive = "{{fountain_cascade_enable_map_fintr_positive}}",
          enable_multi_duration = "{{fountain_cascade_fintr_enable_multi_duration}}",
          fintr_duration_max_value = "{{fountain_cascade_fintr_duration_max_value}}",
          fintr_duration_power_weight = "{{fountain_cascade_fintr_duration_power_weight}}",
          fintr_duration_offset = "{{fountain_cascade_fintr_duration_offset}}",
          fintr_duration_value_upper_bound = "{{fountain_cascade_fintr_duration_value_upper_bound}}",
          duration_ms_attr = "duration_ms",
          fintr_attr = "cascade_ftr_kai",
          save_fintr_duration_to_attr = "cascade_ftr_kai_duration",
          save_fintr_quantile_to_attr = "cascade_ipw_opt_ftr"
        ) \
      .end_if_() \
    .end_if_() \
    .delegate_enrich(
      kess_service = "{{fountain_cascade_wtd_act_predict_kai_kess_service}}",
      recv_item_attrs = [
        {"name":"wtd", "as":"cascade_debias_wtd"},
        {"name":"wtd_percent_10", "as":"cascade_wtd_10p"},
        {"name":"wtd_percent_20", "as":"cascade_wtd_20p"},
        {"name":"wtd_percent_30", "as":"cascade_wtd_30p"},
        {"name":"wtd_percent_40", "as":"cascade_wtd_40p"},
        {"name":"wtd_percent_50", "as":"cascade_wtd_50p"},
        {"name":"wtd_percent_60", "as":"cascade_wtd_60p"},
        {"name":"wtd_percent_70", "as":"cascade_wtd_70p"},
        {"name":"wtd_percent_80", "as":"cascade_wtd_80p"},
        {"name":"wtd_percent_90", "as":"cascade_wtd_90p"},
        ],
      timeout_ms = 100,
      send_item_attrs = ["item_id"], # 这里以后不要再这么写, 需要用自己在下游用 processor 落到 item attr
      send_common_attrs = cascade_ltr_common_feature,
      request_type = "default",
    ) \
    .get_kconf_params(
      skip = "{{skip_fountain_cascade_get_wtd_table}}",
      kconf_configs = [
      {
        "kconf_key": "{{cascade_wtd_score_table_kconf}}",
        "value_type": "json",
        "json_path": "durationSeg",
        "export_common_attr": "cascade_wtd_table_seg"
      },
      {
        "kconf_key": "{{cascade_wtd_score_table_kconf}}",
        "value_type": "json",
        "json_path": "duration_id_0",
        "export_common_attr": "cascade_wtd_table_0"
      },
      {
        "kconf_key": "{{cascade_wtd_score_table_kconf}}",
        "value_type": "json",
        "json_path": "duration_id_1",
        "export_common_attr": "cascade_wtd_table_1"
      },
      {
        "kconf_key": "{{cascade_wtd_score_table_kconf}}",
        "value_type": "json",
        "json_path": "duration_id_2",
        "export_common_attr": "cascade_wtd_table_2"
      },
      {
        "kconf_key": "{{cascade_wtd_score_table_kconf}}",
        "value_type": "json",
        "json_path": "duration_id_3",
        "export_common_attr": "cascade_wtd_table_3"
      },
      {
        "kconf_key": "{{cascade_wtd_score_table_kconf}}",
        "value_type": "json",
        "json_path": "duration_id_4",
        "export_common_attr": "cascade_wtd_table_4"
      },
      {
        "kconf_key": "{{cascade_wtd_score_table_kconf}}",
        "value_type": "json",
        "json_path": "duration_id_5",
        "export_common_attr": "cascade_wtd_table_5"
      },
      {
        "kconf_key": "{{cascade_wtd_score_table_kconf}}",
        "value_type": "json",
        "json_path": "duration_id_6",
        "export_common_attr": "cascade_wtd_table_6"
      },
      {
        "kconf_key": "{{cascade_wtd_score_table_kconf}}",
        "value_type": "json",
        "json_path": "duration_id_7",
        "export_common_attr": "cascade_wtd_table_7"
      },
      {
        "kconf_key": "{{cascade_wtd_score_table_kconf}}",
        "value_type": "json",
        "json_path": "duration_id_8",
        "export_common_attr": "cascade_wtd_table_8"
      },
      {
        "kconf_key": "{{cascade_wtd_score_table_kconf}}",
        "value_type": "json",
        "json_path": "duration_id_9",
        "export_common_attr": "cascade_wtd_table_9"
      },
      ]
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        "cascade_wtd_table_seg",
        "cascade_wtd_table_0",
        "cascade_wtd_table_1",
        "cascade_wtd_table_2",
        "cascade_wtd_table_3",
        "cascade_wtd_table_4",
        "cascade_wtd_table_5",
        "cascade_wtd_table_6",
        "cascade_wtd_table_7",
        "cascade_wtd_table_8",
        "cascade_wtd_table_9",
        "cascade_wtd_duration_mix_threshold",
        "cascade_wtd_duration_mix_max_value"
      ],
      import_item_attr = [
        "cascade_wtd_kai",
        "duration_ms",
        "cascade_wtd_10p",
        "cascade_wtd_20p",
        "cascade_wtd_30p",
        "cascade_wtd_40p",
        "cascade_wtd_50p",
        "cascade_wtd_60p",
        "cascade_wtd_70p",
        "cascade_wtd_80p",
        "cascade_wtd_90p",
      ],
      export_item_attr = [
        "cascade_wtd_percent",
        "cascade_wtd_duration_mix",
      ],
      function_name = "GetMcWtdMix",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(  # duration 和 playtime 阈值的配置, 在 kconf 中配置, 需要保证有序
      import_common_attr = [
        "cascade_wtd_table_seg",
        "cascade_wtd_table_0",
        "cascade_wtd_table_1",
        "cascade_wtd_table_2",
        "cascade_wtd_table_3",
        "cascade_wtd_table_4",
        "cascade_wtd_table_5",
        "cascade_wtd_table_6",
        "cascade_wtd_table_7",
        "cascade_wtd_table_8",
        "cascade_wtd_table_9",
        {"name" : "fountain_mc_wtd_fintr_fintr_low_bound", "as" : "fintr_low_bound"},
        {"name" : "fountain_mc_wtd_fintr_fintr_upper_bound", "as" : "fintr_upper_bound"},
        {"name" : "fountain_mc_wtd_fintr_fintr_power", "as" : "fintr_power"}
      ],
      import_item_attr = [
        "duration_ms",
        "cascade_wtd_kai"
      ],
      export_item_attr = [
        "cascade_wtd_kai_mix",
        "cascade_wtd_fintr"
      ],
      function_name = "GetMcWtdScore",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_lua(
      skip = "{{fountain_skip_calc_cascade_action_once_score}}",
      import_common_attr = [
        "cascade_action_once_watchtime_score_pctr_weight",
        "cascade_action_once_watchtime_score_plvtr_weight",
        "cascade_action_once_watchtime_score_pfintr_quantile_weight",
        "cascade_action_once_watchtime_score_pslide_weight",
        "cascade_action_once_watchtime_score_pwatch_time_weight",
        "cascade_action_once_watchtime_score_pwtd_weight",
        "cascade_action_once_watchtime_score_pwtd_kai_weight",
        "cascade_action_once_watchtime_score_pftr_duration_weight",
        "cascade_action_once_interact_score_pltr_weight",
        "cascade_action_once_interact_score_pwtr_weight",
        "cascade_action_once_interact_score_pftr_weight",
        "cascade_action_once_interact_score_pcmtr_weight",
        "cascade_action_once_interact_score_pcmef_weight",
        "cascade_action_once_interact_score_pptr_weight",
        "cascade_action_once_interact_score_pepstr_weight",
        "cascade_action_once_interact_score_pcltr_weight",
        "cascade_action_once_interact_score_phtr_weight",
      ],
      import_item_attr = [
        "cascade_pctr",
        "cascade_plvtr",
        "cascade_ipw_opt_ftr",
        "cascade_slide_kai",
        "cascade_pwatch_time",
        "cascade_pwtd",
        "cascade_wtd_kai",
        "cascade_ftr_kai_duration",
        "cascade_pltr",
        "cascade_pwtr",
        "cascade_pftr",
        "cascade_pcmtr",
        "cascade_pcestr",
        "cascade_ptr",
        "cascade_pepstr",
        "cascade_pcltr",
        "cascade_phtr"
      ],
      export_item_attr = [
        "cascade_action_once_interact_score",
        "cascade_action_once_watchtime_score",
      ],
      function_for_item = "cal_action_once_score",
      lua_script_file = "./life/cascading/lua/module/fountain_splash_cascading_predict__calc_pxtr.lua",
    ) \
    .if_("fountain_skip_cascade_cluster_id_calc == 0") \
      .transform_item_attr(  # 判断是否是图片
        mappings = [{
          "check_attr_name": "upload_type",
          "check_attr_type": "int",
          "output_attr_name": "is_picture",
          "output_attr_type": "int",
          "rules": [{
            "check_values": [7, 10, 11, 70],
            "output_value": 1
          }]
        }]) \
      .transform_item_attr(  # 判断是否是图片
        mappings = [{
          "check_attr_name": "duration_ms",
          "check_attr_type": "int",
          "output_attr_name": "is_picture",
          "output_attr_type": "int",
          "rules": [{
            "check_range": {
              "upper_bound": 101, # 不包含
            },
            "output_value": 1
          }]
        }]) \
      .switch_("fountain_fast_mc_cluster_method") \
        .case_("duration_quantile") \
          .if_("fountain_variant_mc_enable_gen_living_cluster == 1") \
            .set_attr_value(
              no_overwrite=True,
              item_attrs=[
                {
                  "name": "cascade_cluster_id",
                  "type": "int",
                  "value": 0
                }
              ],
              target_item = {"live_photo_info__is_living": 1}
            ) \
          .end_() \
          .if_("fountain_variant_mc_enable_gen_pic_cluster == 1") \
            .set_attr_value(
              no_overwrite=True,
              item_attrs=[
                {
                  "name": "cascade_cluster_id",
                  "type": "int",
                  "value": 1
                }
              ],
              target_item = {"is_picture": 1}
            ) \
          .end_() \
          .enrich_attr_by_light_function(
            import_common_attr = [
              {"name": "fountain_variant_mc_enable_gen_living_cluster", "as": "enable_gen_living_cluster"},
              {"name": "fountain_variant_mc_enable_gen_pic_cluster", "as": "enable_gen_pic_cluster"},
            ],
            import_item_attr = [
              {"name": "live_photo_info__is_living", "as": "is_living_attr"},
              {"name": "is_picture", "as": "is_picture_attr"},
            ],
            export_item_attr = [
              "photo_cluster_flag", # 除了living和pic之外(可选)的photo, 粗排分桶专用
            ],
            function_name = "SetPhotoClusterFlag",
            class_name = "ExploreLightFunctionSetV2",
          ) \
          .sort(
            score_from_attr = "duration_ms",
            desc = False,
            target_item = {"photo_cluster_flag": 1}
          ) \
          .enrich_attr_by_light_function(
            import_common_attr = [
              {"name": "fountain_mc_time_cluster_num", "as": "explore_mc_time_cluster_num"},
              {"name": "fountain_variant_mc_time_cluster_base_id", "as": "mc_time_cluster_base_id"}
            ],
            export_item_attr = [
              "cascade_cluster_id",
            ],
            function_name = "EqualSizeCluster",
            class_name = "ExploreLightFunctionSetV2",
            target_item = {"photo_cluster_flag": 1}
          ) \
        .case_("random_cluster") \
          .shuffle() \
          .enrich_attr_by_light_function(
            import_common_attr = [
              {"name": "fountain_mc_random_cluster_num", "as": "cluster_num"},
              {"name": "fountain_mc_random_cluster_base_id", "as": "cluster_base_id"},
              {"name": "fountain_mc_random_cluster_enable_diversity_cluster", "as": "enable_diversity_cluster"},
              {"name": "fountain_mc_random_cluster_diversity_ratio", "as": "diversity_ratio"},
              {"name": "fountain_mc_random_cluster_enable_diversity_cluster_hetu_level_two", "as": "enable_diversity_cluster_hetu_level_two"},
            ],
            import_item_attr = [
              {"name": "hetu_tag_level_info_v2__hetu_level_one", "as": "hetu_level_one"},
              {"name": "hetu_tag_level_info_v2__hetu_level_two", "as": "hetu_level_two"},
            ],
            export_item_attr = [
              "cascade_cluster_id",
            ],
            function_name = "RandomCluster",
            class_name = "ExploreLightFunctionSetV2"
          ) \
        .default_() \
          .pack_common_attr(
            input_common_attrs = ["similar_user_colossus_hetu_list","explore_hetu_list"],
            output_common_attr = "input_explore_interest_hetu_list",
          ) \
          .if_("enable_user_explore_interest_cluster_from_list==1") \
            .enrich_attr_by_lua(
              import_common_attr = [
                "similar_user_colossus_hetu_list",
                "explore_hetu_list",
                "input_explore_interest_hetu_list"
              ],
              export_common_attr = [
                "similar_user_list_size",
                "explore_history_hetu_list_size",
                "explore_hetu_list_all_size"
              ],
              function_for_common = "cascade_explore_list_size",
              lua_script_file = "./life/cascading/lua/module/fountain_splash_cascading_predict__control.lua",
            ) \
            .perflog_attr_value(
              check_point="cascade_explore_list_size",
              common_attrs=["similar_user_list_size","explore_history_hetu_list_size","explore_hetu_list_all_size"],
              aggregator="avg",
            ) \
          .end_if_() \
          .explore_rule_cluster_enricher(
            save_cluster_id_to_attr = "cascade_cluster_id",
            check_point = "cascade_v2",
            enable_time_cluster = "{{fountain_mc_enable_time_cluster}}",
            enable_living_cluster = "{{fountain_mc_enable_living_cluster}}",
            enable_user_interest_level_one_cluster = "{{fountain_cascade_interest_use_level_one}}",
            enable_hetu_cluster = "{{fountain_mc_enable_hetu_cluster}}",
            enable_merge_interact_cluster = "{{fountain_mc_enable_merge_interact_cluster_v2}}",
            enable_interact_cluster = "{{fountain_mc_enable_interact_cluster}}",
            input_short_interest_attr ="short_interest",
            input_action_interest_attr ="action_interest",
            input_long_interest_attr ="long_interest",
            input_random_explore_interest_attr ="random_explore_interest",
            enable_user_explore_interest_cluster = "{{enable_user_explore_interest_cluster}}",
            explore_interest_reason = "{{explore_interest_reason_str}}",
            enable_explore_use_hetu_level_one = "{{enable_explore_use_hetu_level_one}}",
            explore_interest_cnt = "{{explore_interest_cnt}}",
            duration_cluster_cfg_str = "{{fountain_cascade_duration_cluster_cfg}}",
            hetu_level_one_attr = "hetu_tag_level_info__hetu_level_one",
            hetu_level_two_attr = "hetu_tag_level_info__hetu_level_two",
            enable_user_explore_interest_cluster_from_list = "{{enable_user_explore_interest_cluster_from_list}}",
            input_explore_interest_attr = "input_explore_interest_hetu_list",
            duration_ms_attr = "duration_ms",
            is_picture_attr = "is_picture",
            is_living_attr = "live_photo_info__is_living",
            enable_user_follow_author_cluster = "{{enable_user_follow_author_cluster}}",
            enable_user_follow_author_cluster_first = "{{enable_user_follow_author_cluster_first}}",
            is_follow_author_attr = "is_photo_author_followed",
            enable_hot_content_cluster = "{{enable_fountain_mc_hot_content_cluster}}",
            hot_content_exp_tag_attr = "hot_content_thompson_sampling_exp_tag_list",
          ) \
      .end_() \
      .if_("fountain_skip_long_term_interest_ee_cascade==0") \
        .calc_long_term_interest_ee_score(
          user_info_pb_name = "user_info_ptr",
          hetu_attrs = "hetu_tag_level_info__hetu_level_one;hetu_tag_level_info__hetu_level_two;hetu_tag_level_info__hetu_level_three;hetu_tag_level_info__hetu_level_four;hetu_tag_level_info__hetu_face_id;hetu_tag_level_info__hetu_tag",
          enable_click_history = "{{fountain_mc_enable_click_history}}",
          enable_like_history = "{{fountain_mc_enable_like_history}}",
          enable_follow_history = "{{fountain_mc_enable_follow_history}}",
          enable_long_view_history = "{{fountain_mc_enable_long_view_history}}",
          long_view_threshold = "{{fountain_mc_long_view_threshold}}",
          export_item_attr = "cascade_long_term_interest_ee_score",
          enable_division_way = "{{fountain_mc_enable_division_way}}",
          photo_hetu_tag_level_info_type = "{{foutnain_mc_photo_hetu_tag_level_info_type}}",
          boost_threshold = "{{fountain_mc_long_term_interest_ee_boost_threshold}}",
        ) \
        .get_kconf_params(
          kconf_configs = [{
            "kconf_key": "reco.fountain.highValueHetuList",
            "value_type": "list_int64",
            "defult_value": [134, 120, 114, 189, 220, 316, 179, 199, 325, 161, 208, 203],
            "export_common_attr": "high_value_hetu_list"
          }]
        ) \
        .pack_item_attr(
          item_source = {
            "reco_results": True,
          },
          mappings = [{
            "from_item_attr": "hetu_level_one_v2_index_cascade",
            "to_common_attr": "hetu_level_one_v2_index_cascade_list_no_dedup",
            "dedup_to_common_attr": False,
          }],
        ) \
        .enrich_attr_by_lua(
          import_common_attr = [
            "fountain_mc_high_value_hetu_debias_coef",
            "fountain_mc_enable_only_longterm_debias",
            "high_value_hetu_list",
            "fountain_mc_enable_lt_weight_adjust",
            "hetu_level_one_v2_index_cascade_list_no_dedup",
            "fountain_fast_ensemble_weight_cascade_long_term_interest_ee_score",
            "fountain_mc_lt_weight_adjust_threshold",
            "fountain_mc_lt_weight_adjust_coef",
          ],
          import_item_attr = [
            "cascade_long_term_interest_ee_score",
            "hetu_tag_level_info_v2__hetu_level_one",
            ],
          export_item_attr = [
            "cascade_long_term_interest_ee_score",
          ],
          export_common_attr = [
            "fountain_fast_ensemble_weight_cascade_long_term_interest_ee_score"
          ],
          function_for_item = "calc_mc_high_value_hetu_debias",
          function_for_common = "calc_mc_max_hetu_one_rate",
          lua_script_file = "./life/cascading/lua/module/fountain_fast_cascading_score__high_value_hetu_debias.lua",
        ) \
        .log_debug_info(
          common_attrs = [
            "page",
            "high_value_hetu_list",
            "fountain_fullrank_high_value_hetu_debias_coef",
            "hetu_level_one_v2_index_cascade_list_no_dedup",
            "fountain_fast_ensemble_weight_cascade_long_term_interest_ee_score",
          ],
          item_attrs = [
            "photo_id",
            "cascade_long_term_interest_ee_score",
            "hetu_tag_level_info_v2__hetu_level_one",
          ],
          item_num_limit = 10,
          for_debug_request_only = True,
        ) \
      .end_if_() \
      .if_("fountain_mc_s1_enable_hetu_cluster_adjust_cut_ratio == 1 or fountain_mc_s1_enable_duration_cluster_adjust_hetu_score == 1") \
        .explore_photo_distribution_adjust_enricher(
          colossus_total_count_attr = "colossus_hetu_distribution_total_count",
          user_hetu_stat_attr = "colossus_hetu_distribution_hetu_stat",
          colossus_total_count_threshold = "{{fountain_mc_hetu_distribution_colossus_total_count_threshold}}",
          max_count = "{{fountain_mc_s1_hetu_distribution_stat_max_count}}",
          global_fuse_corr = "{{fountain_mc_s1_hetu_distribution_global_fuse_corr}}",
          enable_daily_update_global_distribution = "{{fountain_enable_daily_update_global_hetu_distribution}}",
          latest_global_hetu_distribution_attr = "fountain_latest_global_hetu_distribution_map",
          global_hetu_stat_redis_key_prefix = "{{fountain_global_hetu_stat_redis_key_prefix}}",
          global_hetu_distribution_use_fountain_flag = True,
          hetu_level_one_attr = "hetu_tag_level_info__hetu_level_one",
          candidate_hetu_adjust_coeff_map_attr = "fountain_mc_s1_candidate_hetu_adjust_coeff_map" # 候选集调整系数
        ) \
      .end_() \
      .if_("fountain_mc_s1_enable_hetu_cluster_adjust_cut_ratio == 1") \
        .split_string(
          input_common_attr="fountain_mc_s1_hetu_cluster_hetu_adjust_paras",
          output_common_attr="mc_s1_hetu_cluster_hetu_adjust_para_list",
          delimiters=",",
          parse_to_double=True,
        ) \
      .end_() \
      .if_("fountain_mc_s1_enable_duration_cluster_adjust_hetu_score == 1") \
        .split_string(
          input_common_attr="fountain_mc_s1_duration_cluster_hetu_adjust_paras",
          output_common_attr="mc_s1_duration_cluster_hetu_adjust_para_list",
          delimiters=",",
          parse_to_double=True,
        ) \
      .end_() \
      .set_attr_value(
        item_attrs = [
          {
            "name": "mc_adjust_coeff_final",
            "type": "double",
            "value": 1.0
          }
        ]
      ) \
      .if_("life_fountain_enable_mc_marketing_compensation_adjust == 1") \
        .mc_marketing_compensation_adjust() \
      .end_() \
      .if_("enable_life_fountain_mc_low_cost_photo_adjust == 1") \
        .mc_low_cost_photo_adjust() \
      .end_() \
      .if_("enable_life_fountain_mc_llm_negative_photo_adjust == 1") \
        .mc_llm_negative_photo_adjust() \
      .end_() \
      .if_("fountain_fast_cascade_enable_cluster_variant_sort == 1") \
        .explore_cluster_variant_sort_enricher(
          check_point = "cascade",
          cluster_sort_list_attr_name = "cascade_cluster_id",
          cluster_config = "{{fountain_combine_variant_cluster_sort_config}}",
          global_cut_ratio = "{{fountain_cascade_variant_cluster_global_cut_ratio}}",
          min_survival = "{{fountain_cascade_variant_cluster_min_survival}}",
          enable_proportional = "{{fountain_cascade_variant_cluster_sort_enable_proportional}}",
          size_limit = "{{fountain_cascade_variant_cluster_sort_size_limit}}",
          use_power_calc = "{{fountain_cascade_variant_cluster_sort_use_power_calc}}",
          use_power_calc_v2 = "{{fountain_cascade_variant_cluster_sort_use_power_calc_v2}}",
          rank_value_fusion_type = "{{fountain_cascade_variant_cluster_sort_rank_value_fusion_type}}",
          rank_smooth = "{{fountain_cascade_variant_cluster_sort_rank_smooth}}",
          use_reciprocal = "{{fountain_cascade_variant_cluster_sort_use_reciprocal}}",
          user_info_ptr_attr = "user_info_ptr",
          action_day = "{{fountain_mc_variant_weight_action_day_num}}",
          time_cluster_base_id = "{{fountain_variant_mc_time_cluster_base_id}}",
          queues = explore_cluster_sort_queues,
          fixed_final_size="{{fountain_mc_cluster_fixed_final_size}}",
          enable_dynamic_cut_ratio="{{fountain_mc_cluster_enable_dynamic_cut_ratio}}",
          save_score_to_attr = "cascade_variant_sort_score",
          save_filter_flag_to_attr = "cascade_s1_filter_flag",  # 用于标记是否截断 = 1 时要被截断, 缺省或其他值不截断
          # 根据兴趣分布调整兴趣分桶截断比例和时长分桶内 score
          enable_hetu_cluster_adjust_cut_ratio  = "{{fountain_mc_s1_enable_hetu_cluster_adjust_cut_ratio}}",
          enable_duration_cluster_adjust_hetu_score  = "{{fountain_mc_s1_enable_duration_cluster_adjust_hetu_score}}",
          hetu_cluster_hetu_adjust_paras_attr = "mc_s1_hetu_cluster_hetu_adjust_para_list",
          duration_cluster_hetu_adjust_para_attr = "mc_s1_duration_cluster_hetu_adjust_para_list",
          duration_cluster_enable_unknown_hetu_adjust = "{{fountain_mc_s1_duration_cluster_enable_unknown_hetu_adjust}}",
          candidate_hetu_adjust_coeff_map_attr = "fountain_mc_s1_candidate_hetu_adjust_coeff_map",
          hetu_level_one_attr = "hetu_tag_level_info__hetu_level_one",
          enable_ceil_keep_size = "{{fountain_mc_s1_enable_ceil_keep_size}}",
          enable_hetu_adjust_fixed_size = "{{fountain_mc_s1_enable_hetu_adjust_fixed_size}}",
          save_cluster_id_common_attr = "mc_s1_cluster_id",
          save_cluster_cnt_common_attr = "mc_s1_cluster_cnt",
          vv_min_rank_pow_weight = "{{fountain_mc_vv_min_rank_pow_weight}}",
          act_min_rank_pow_weight = "{{fountain_mc_act_min_rank_pow_weight}}",
          play_min_rank_pow_weight = "{{fountain_mc_play_min_rank_pow_weight}}",
          combine_min_rank_pow_weight = "{{fountain_mc_combine_min_rank_pow_weight}}",
          enable_vv_min_rank = "{{fountain_mc_enable_vv_min_rank}}",
          enable_act_min_rank = "{{fountain_mc_enable_act_min_rank}}",
          enable_play_min_rank = "{{fountain_mc_enable_play_min_rank}}",
          enable_combine_min_rank = "{{fountain_mc_enable_combine_min_rank}}",
          vv_min_rank_weight = "{{fountain_mc_vv_min_rank_weight}}",
          act_min_rank_weight = "{{fountain_mc_act_min_rank_weight}}",
          play_min_rank_weight = "{{fountain_mc_play_min_rank_weight}}",
          enable_score_adjust = "{{life_fountain_enable_mc_score_adjust}}",
          adjust_coeff_final_attr = "mc_adjust_coeff_final",
        ) \
        .if_("fountain_enable_cascade_distill_full_link_sample == 1") \
          .get_kconf_params(
            kconf_configs=[
              {
                "kconf_key": "reco.offline.fountainFulllinkDistillRankParam",
                "value_type": "json",
                "json_path": "cascade_s1_full_link_distill_sample_begin",
                "export_common_attr": "fountain_cascade_s1_full_link_distill_sample_begin"
              },
              {
                "kconf_key": "reco.offline.fountainFulllinkDistillRankParam",
                "value_type": "json",
                "json_path": "cascade_s1_full_link_distill_sample_end",
                "export_common_attr": "fountain_cascade_s1_full_link_distill_sample_end"
              },
              {
                "kconf_key": "reco.offline.fountainFulllinkDistillRankParam",
                "value_type": "json",
                "json_path": "cascade_s1_full_link_distill_sample_num",
                "export_common_attr": "fountain_cascade_s1_full_link_distill_sample_num"
              },
              {
                "kconf_key": "reco.offline.fountainFulllinkDistillRankParam",
                "value_type": "json",
                "json_path": "cascade_s1_full_link_distill_sample_ratio",
                "export_common_attr": "fountain_cascade_s1_full_link_distill_sample_ratio"
              },
            ]
          ) \
          .explore_full_link_context_sample_reco_log_enricher(
            target_item = { "cascade_s1_filter_flag": 1},
            sample_config = [
              {
                "sample_begin": "fountain_cascade_s1_full_link_distill_sample_begin",
                "sample_end": "fountain_cascade_s1_full_link_distill_sample_end",
                "sample_num": "fountain_cascade_s1_full_link_distill_sample_num",
                "label_name": "cas_neg_stage1",
              },
            ],
            sample_ratio = "fountain_cascade_s1_full_link_distill_sample_ratio",
            user_info_attr = "user_info_ptr",
            enable_set_user_info = True,
            output_attr = "fountain_full_link_reco_log_message",
            cascade_pctr = "cascade_pctr",
            cascade_pltr = "cascade_pltr",
            cascade_pwtr = "cascade_pwtr",
            cascade_pftr = "cascade_pftr",
            cascade_pptr = "cascade_ptr",
            cascade_pcmtr = "cascade_pcmtr",
            cascade_plvtr = "cascade_plvtr",
            cascade_pvtr = "cascade_pwatch_time",
          ) \
        .end_if_() \
        .filter_by_attr(
          attr_name = "cascade_s1_filter_flag",
          remove_if = "==",
          compare_to = 1,
          remove_if_attr_missing = False,
        ) \
      .end_if_() \
    .end_if_()