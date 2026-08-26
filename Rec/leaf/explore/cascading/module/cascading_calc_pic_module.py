from cascading import CommonModule

class CascadingCalcPicModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .if_(
        # 采样请求已计算过 picture ，不再计算，v2 计算方式推全时应替换
        "not _IS_PERF_SAMPLING_REQUEST_ or _IS_PERF_SAMPLING_REQUEST_ == 0") \
        .if_("enable_calc_is_picture_v2 > 0", to_be_delete = "date=2023-11-16;committer=lihengchao") \
          .enrich_attr_by_light_function(
            import_item_attr = [
              "upload_type",
              "duration_ms",
              "picture_type"
            ],
            export_item_attr = [
              "is_picture",
            ],
            function_name = "IsPictureV2",
            class_name = "ExploreLightFunctionSetV2",
          ) \
        .else_() \
          .enrich_attr_by_light_function(
            import_item_attr = [
              "upload_type",
              "duration_ms",
            ],
            export_item_attr = [
              "is_picture",
            ],
            function_name = "IsPicture",
            class_name = "ExploreLightFunctionSetV2",
          ) \
        .end_() \
      .end_() \
      .if_("enable_cascading_define_longpic_picset == 1", to_be_delete = "date=2024-05-29;committer=liuyanlei") \
        .enrich_attr_by_light_function(
          import_item_attr = [
            "upload_type",
            "duration_ms",
            "picture_type"
          ],
          export_item_attr = [
            {"name": "is_picture", "as": "is_longpic_picset"}
          ],
          function_name = "IsLongPicAndPicSet",
          class_name = "ExploreLightFunctionSetV2",
          target_item={
            "is_picture": 1
          }
        ) \
        .dispatch_common_attr(
          from_common_attr="longpic_picset_score_fixed",
          to_item_attr="longpic_picset_score",
          target_item={
            "is_longpic_picset": 1
          }
        ) \
      .end_if_() \
      .if_("enable_request_explore_pic_hive_index == 1 or enable_request_explore_pic_hive_index_v2 == 1"
        + "  or enable_explore_pic_revisited_item == 1") \
        .get_item_attr_by_remote_index(
          kess_service = "{{explore_picture_hive_index_kess_service}}",
          timeout_ms = 100,
          partition_size = 1000,
          attrs = [
            "is_boost_hetu_pic",
            "revisited_rate_1d",
            "revisited_rate_3d",
            "revisited_rate_7d"
          ],
          target_item = {"is_picture": 1},
        ) \
      .end_() \
      .if_("enable_calc_pic_target_hetu_photo == 1 or enable_calc_pic_target_hetu_photo_v2 == 1 or enable_explore_pic_revisited_item == 1 " +
             " or enable_calc_pic_target_high_interact == 1") \
        .enrich_attr_by_light_function(
          target_item = {"is_picture": 1},
          import_common_attr= [
            {"name": "pic_target_hetu_set", "as": "attr_list"},
          ],
          import_item_attr = [
            {"name": "hetu_tag_level_info__hetu_level_one", "as": "attrs"}
          ],
          export_item_attr = [
            {"name": "is_in_set", "as": "is_key_target_hetu_pic"}
          ],
          function_name = "AttrListIsInSet",
          class_name = "ExploreLightFunctionSetV2"
        ) \
      .end_() \
      .if_("enable_calc_xhs_type_picture_v2 == 1 and enable_calc_xhs_type_picture == 0") \
        .enrich_attr_by_light_function(  # 替换prerank后的cal_xhs_type_picture
          target_item = { "is_picture" : 1 },
          import_common_attr = [
            "xhs_hetu_set",
            "xhs_hetu_memorydata_set",
            "calc_xhs_type_mode",
          ],
          import_item_attr = [
            "hetu_tag_level_info__hetu_level_two",
            "hetu_tag_level_info__hetu_level_three",
            "hetu_tag_level_info__hetu_level_four",
          ],
          export_item_attr = [
            "is_xhs_type_photo"
          ],
          function_name = "IsXhsTypePhoto",
          class_name = "ExploreLightFunctionSetV2",  
        ) \
      .end_() \
      .if_("(enable_explore_calc_pic_interest_cluster == 1 or enable_explore_calc_pic_cluster_id_632 == 1) and enable_explore_calc_photo_cluster_id_632 == 0") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "remap_cluster_id_632_list",
          ],
          import_item_attr = [
            "hetu_sim_cluster_id",
          ],
          export_item_attr = [
            "cluster_id_632",
          ],
          function_name = "CalcClusterId632",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {"is_picture": 1}
        ) \
      .end_() \
      .if_("explore_enable_user_pic_growth_cluster_boost == 1 and ((uDoubleOutsideValidPicClusterCnt7dKV or 0) < explore_user_pic_growth_cluster_boost_interest_thresh)") \
        .enrich_attr_by_light_function(
          target_item = {"is_picture": 1},
          import_common_attr= [{"name": "uPicGrowthCidList", "as": "attr_list"},],
          import_item_attr = [{"name": "cluster_id_632", "as": "attr"}],
          export_item_attr = [{"name": "is_in_set", "as": "is_pic_growth_cluster"}],
          function_name = "AttrIsInSet",
          class_name = "ExploreLightFunctionSetV2"
        ) \
      .end_() \
      .if_("enable_explore_calc_pic_interest_cluster == 1") \
        .split_string(
          input_common_attr = "explore_pic_interest_cluster_target_hetu_str_v2",
          output_common_attr = "explore_pic_interest_cluster_target_hetu_list_v2",
          delimiters = ",",
          trim_spaces = True,
          skip_empty_tokens = True,
          parse_to_int = True
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "uPicValidInterestClusterIdList",
            "uPicLongInterestClusterIdList",
            {"name": "explore_pic_interest_cluster_limit_hetu", "as": "limit_hetu"},
            {"name": "explore_pic_interest_cluster_target_hetu_list_v2", "as": "target_hetu_list_v2"},
          ],
          import_item_attr = [
            "cluster_id_632",
            "hetu_tag_level_info_v2__hetu_level_one",
          ],
          export_item_attr = [
            "is_pic_valid_interest_cluster",
            "is_pic_long_interest_cluster",
          ],
          function_name = "CalcPicInterestCluster",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {"is_picture": 1}
        ) \
      .end_() \
      .if_("enable_explore_calc_pic_search_cluster_score == 1") \
        .split_string(
          input_common_attr = "explore_pic_search_cluster_target_hetu_str_v2",
          output_common_attr = "explore_pic_search_cluster_target_hetu_list_v2",
          delimiters = ",",
          trim_spaces = True,
          skip_empty_tokens = True,
          parse_to_int = True
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "uPicSearchInterestClusterIdList",
            "uPicSearchInterestClusterScoreList",
            {"name": "explore_pic_search_cluster_limit_hetu", "as": "limit_hetu"},
            {"name": "explore_pic_search_cluster_target_hetu_list_v2", "as": "target_hetu_list_v2"},
          ],
          import_item_attr = [
            "cluster_id_632",
            "hetu_tag_level_info_v2__hetu_level_one",
          ],
          export_item_attr = [
            "pic_search_interest_cluster_score",
          ],
          function_name = "CalcPicSearchClusterScore",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {"is_picture": 1}
        ) \
      .end_() \
      .if_("enable_explore_calc_pic_user_long_interest_hetu == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "uHetuCategoryInterestlv1IdList",
            "uHetuCategoryInterestlv1ScoreList",
            {"name": "explore_pic_user_long_interest_hetu_score_thresh", "as": "score_threshold"}
          ],
          export_common_attr = [
            {"name": "output_hetu_id_list", "as": "pic_user_long_interest_hetu_list"}
          ],
          function_name = "GetUserPicLongInterestHetu",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {"is_picture": 1}
        ) \
      .end_() \
      .if_("enable_explore_pic_search_candicate_expand_by_cluster == 1") \
        .switch_("explore_pic_search_cluster_calc_mode") \
          .case_(1) \
            .pack_item_attr(
              item_source = {
                "reco_results": True
              },
              mappings = [{
                "from_item_attr": "cluster_id_632",
                "to_common_attr": "pic_search_cluster_id_632_list",
                "aggregator": "concat"
              }],
              target_item = {"is_pic_search": 1}
            ) \
            .if_("#(user_recent_search_valid_play_pid_list or {}) > 0") \
              .enrich_attr_by_light_function(
                item_list_from_attr = "user_recent_search_valid_play_pid_list", # user_search_valid_play_pid_list 的子序列，已经填充过 hetu_sim_cluster_id
                export_item_attr = [
                  "hetu_sim_cluster_id",
                ],
                function_name = "EmptyFunction",
                class_name = "ExploreLightFunctionSetV2",
              ) \
              .enrich_attr_by_light_function(
                item_list_from_attr = "user_recent_search_valid_play_pid_list",
                import_common_attr = [
                  "remap_cluster_id_632_list",
                ],
                import_item_attr = [
                  "hetu_sim_cluster_id",
                ],
                export_item_attr = [
                  "cluster_id_632",
                ],
                function_name = "CalcClusterId632",
                class_name = "ExploreLightFunctionSetV2",
              ) \
              .if_("enable_explore_pic_recent_search_frequency_control == 1") \
                .enrich_attr_by_light_function(
                  import_common_attr = [
                    "standard_explore_realshow_pid_list",
                    "uStandardExploreRealshowTimestampList",
                    "uStandardRealShowPicAllIdList",
                  ],
                  export_common_attr = [
                    "explore_pic_realshow_pid_list",
                    "explore_pic_realshow_timestamp_list",
                  ],
                  function_name = "GetExplorePicRealshowList",
                  class_name = "ExploreLightFunctionSetV2",
                ) \
                .enrich_attr_by_light_function(
                  item_list_from_attr = "explore_pic_realshow_pid_list", # standard_explore_realshow_pid_list 的子序列，已经填充过 cluster_id_632
                  export_item_attr = [
                    "cluster_id_632",
                  ],
                  function_name = "EmptyFunction",
                  class_name = "ExploreLightFunctionSetV2",
                ) \
                .pack_item_attr(
                  item_source = {
                    "common_attr": ["explore_pic_realshow_pid_list"],
                  },
                  mappings = [{
                    "from_item_attr": "cluster_id_632",
                    "to_common_attr": "explore_pic_realshow_cluster_id_632_list",
                    "aggregator": "concat",
                    "default_val": 0
                  }],
                ) \
                .enrich_attr_by_light_function(
                  item_list_from_attr = "user_recent_search_valid_play_pid_list",
                  import_common_attr = [
                    "user_recent_search_valid_play_timestamp_list",
                    "explore_pic_realshow_cluster_id_632_list",
                    "explore_pic_realshow_timestamp_list",
                    "explore_pic_recent_search_cid_max_realshow_cnt"
                  ],
                  import_item_attr = [
                    "cluster_id_632",
                  ],
                  export_common_attr = [
                    "pic_recent_search_cluster_id_632_list",
                  ],
                  function_name = "GetExplorePicRecentSearchFrequencyControlCidList",
                  class_name = "ExploreLightFunctionSetV2",
                ) \
              .else_() \
                .pack_item_attr(
                  item_source = {
                    "reco_results": False,
                    "common_attr": ["user_recent_search_valid_play_pid_list"],
                  },
                  mappings = [{
                    "from_item_attr": "cluster_id_632",
                    "to_common_attr": "pic_recent_search_cluster_id_632_list",
                    "aggregator": "concat",
                    "dedup_to_common_attr": True
                  }],
                ) \
              .end_() \
            .end_() \
          .default_() \
            .pack_item_attr(
              item_source = {
                "reco_results": True
              },
              mappings = [{
                "from_item_attr": "cluster_id_632",
                "to_common_attr": "pic_search_cluster_id_632_list",
                "aggregator": "concat"
              }],
              target_item = {"is_pic_search": 1}
            ) \
            .pack_item_attr(
              item_source = {
                "reco_results": True
              },
              mappings = [{
                "from_item_attr": "cluster_id_632",
                "to_common_attr": "pic_recent_search_cluster_id_632_list",
                "aggregator": "concat"
              }],
              target_item = {"is_pic_recent_search": 1}
            ) \
        .end_() \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "pic_search_cluster_id_632_list", "as": "attr_list"},
          ],
          import_item_attr = [
            {"name": "cluster_id_632", "as": "attr"}
          ],
          export_item_attr = [
            {"name": "is_in_set", "as": "is_pic_search_cluster"}
          ],
          function_name = "AttrIsInSet",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {"is_picture": 1},
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "pic_recent_search_cluster_id_632_list", "as": "attr_list"},
          ],
          import_item_attr = [
            {"name": "cluster_id_632", "as": "attr"}
          ],
          export_item_attr = [
            {"name": "is_in_set", "as": "is_pic_recent_search_cluster"}
          ],
          function_name = "AttrIsInSet",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {"is_picture": 1},
        ) \
      .end_() \
      .if_("enable_explore_calc_pic_double_valid_interest_cluster == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "uDoubleOutsideValidPicCluster7dList", "as": "attr_list"},
          ],
          import_item_attr = [
            {"name": "cluster_id_632", "as": "attr"}
          ],
          export_item_attr = [
            {"name": "is_in_set", "as": "is_pic_double_valid_interest_cluster"}
          ],
          function_name = "AttrIsInSet",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {"is_picture": 1},
        ) \
      .end_() \
      .if_("enable_explore_calc_pic_single_valid_interest_cluster == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "uSingleValidPicCluster7dList", "as": "attr_list"},
          ],
          import_item_attr = [
            {"name": "cluster_id_632", "as": "attr"}
          ],
          export_item_attr = [
            {"name": "is_in_set", "as": "is_pic_single_valid_interest_cluster"}
          ],
          function_name = "AttrIsInSet",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {"is_picture": 1},
        ) \
      .end_() \
      .if_("enable_explore_calc_pic_recent_interest_cluster == 1") \
        .enrich_attr_by_light_function(
          item_list_from_attr = "explore_recent_interest_colossus_trigger_list",
          import_common_attr = [
            "remap_cluster_id_632_list",
          ],
          import_item_attr = [
            "hetu_sim_cluster_id",
          ],
          export_item_attr = [
            "cluster_id_632",
          ],
          function_name = "CalcClusterId632",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("enable_explore_calc_pic_double_outside_valid_interest_num == 1") \
        .gen_common_attr_by_lua(
          attr_map = {
            "pic_double_outside_valid_interest_num": "#(uDoubleOutsideValidPicCluster7dList or {})",
          }
        ) \
      .end_() \
      .if_("enable_explore_calc_pic_recent_interest_cluster_score == 1") \
        .enrich_attr_by_light_function(
          item_list_from_attr = "explore_recent_interest_colossus_trigger_list",
          import_common_attr = [
            {"name": "enable_explore_calc_recent_interest_cluster_score_weight_decay", "as": "enable_weight_decay"},
            {"name": "explore_calc_recent_interest_cluster_score_weight_decay_power", "as": "weight_decay_power"},
            {"name": "enable_explore_calc_recent_interest_cluster_score_click_num_thres", "as": "enable_click_num_thres"},
            {"name": "explore_recent_interest_colossus_trigger_weight_list", "as": "weight_list"},
          ],
          import_item_attr = [
            "cluster_id_632",
          ],
          export_common_attr = [
            {"name": "interest_cluster_id_list", "as": "explore_pic_recent_interest_cluster_id_list"},
            {"name": "interest_cluster_score_list", "as": "explore_pic_recent_interest_cluster_score_list"},
          ],
          function_name = "GetPicRecentInterestScoreList",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("enable_explore_calc_pic_interest_cid_collaborative_score == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "pic_interest_cid_collaborative_score_map", "as": "c2c_score_map"},
            {"name": "uDoubleOutsideValidPicCluster7dList", "as": "explore_interest_cluster_id_list"},
            {"name": "uPicLongInterestClusterIdList", "as": "long_interest_cluster_id_list"},
            {"name": "enable_explore_pic_calc_u2c_by_explore_pic_interest", "as": "calc_u2c_by_explore_pic_interest"},
            {"name": "enable_explore_pic_calc_u2c_by_long_interest", "as": "calc_u2c_by_long_interest"},
            {"name": "enable_explore_pic_u2c_only_low_interest", "as": "u2c_only_low_interest"},
            {"name": "uDoubleOutsideValidPicClusterCnt7dKV", "as": "explore_pic_interest_num"},
            {"name": "explore_pic_u2c_low_interest_threshold", "as": "u2c_low_interest_threshold"},
            {"name": "enable_explore_pic_interest_cid_collaborative_set_default_value", "as": "enable_interest_cid_set_default_value"},
            {"name": "explore_pic_interest_cid_collaborative_default_value", "as": "interest_cid_default_value"},
          ],
          import_item_attr = [
            {"name": "cluster_id_632", "as": "cluster_id"},
          ],
          export_item_attr = [
            {"name": "collaborative_score", "as": "pic_u2c_collaborative_score"}
          ],
          function_name = "CalPicInterestCidCollaborativeScore",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {"is_picture": 1},
        ) \
      .end_() \
      .if_("enable_explore_cal_pic_career_interest_tagnex_tgi == 1") \
        .enrich_attr_by_light_function(
          target_item = {"is_picture": 1},
          import_common_attr = [
            {"name": "explore_user_pic_career_interest_tagnex_tgi_list", "as": "match_list"},
            {"name": "explore_pic_career_interest_tagnex_tgi_coeff", "as": "coeff"},
            {"name": "explore_pic_career_interest_tagnex_tgi_bias", "as": "bias"},
            {"name": "explore_pic_career_interest_tagnex_circle_attr_min", "as": "attr_min"},
            {"name": "explore_pic_career_interest_tagnex_circle_attr_max", "as": "attr_max"},
            {"name": "explore_pic_career_interest_tagnex_circle_use_single_match_item", "as": "use_single_match_item"},
          ],
          import_item_attr = [
            {"name": "hetu_tag_level_info__hetu_tag", "as": "hetu_tag"},
          ],
          export_item_attr = [
            {"name": "match_score", "as": "pic_career_interest_tagnex_tgi_score"}
          ],
          function_name = "CalMatchScore",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("enable_explore_cal_pic_age_interest_tagnex_tgi == 1") \
        .enrich_attr_by_light_function(
          target_item = {"is_picture": 1},
          import_common_attr = [
            {"name": "explore_user_pic_age_interest_tagnex_tgi_list", "as": "match_list"},
            {"name": "explore_pic_age_interest_tagnex_tgi_coeff", "as": "coeff"},
            {"name": "explore_pic_age_interest_tagnex_tgi_bias", "as": "bias"},
            {"name": "explore_pic_age_interest_tagnex_circle_attr_min", "as": "attr_min"},
            {"name": "explore_pic_age_interest_tagnex_circle_attr_max", "as": "attr_max"},
            {"name": "explore_pic_age_interest_tagnex_circle_use_single_match_item", "as": "use_single_match_item"},
          ],
          import_item_attr = [
            {"name": "hetu_tag_level_info__hetu_tag", "as": "hetu_tag"},
          ],
          export_item_attr = [
            {"name": "match_score", "as": "pic_age_interest_tagnex_tgi_score"}
          ],
          function_name = "CalMatchScore",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_()
