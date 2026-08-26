#!/usr/bin/env python3
# coding=utf-8

from dragonfly.common_leaf_dsl import LeafFlow
from dragonfly.ext.explore.explore_api_mixin import ExploreApiMixin
from dragonfly.ext.subdivision.subdivision_api_mixin import subdivisionApiMixin
from dragonfly.ext.embed_calc.embed_calc_api_mixin import EmbedCalcApiMixin
from dragonfly.ext.embedding.embedding_api_mixin import EmbeddingApiMixin
from filter.filter_base_flow import FilterBaseFlow
from util import enrich_ab_param


class FilterSplashV2Flow(FilterBaseFlow, subdivisionApiMixin, EmbedCalcApiMixin, ExploreApiMixin, EmbeddingApiMixin):
  _ITEM_ATTR_MAP = {
    "explore_server_show_attr": "explore_stat__show_count",
  }

  _FILTERS = [
    {
      "name": "source_dup_content_id_filter",
      "enable": "{{fountain_enable_source_content_filter}}",
      "source_pid_attr": "featureSourcePId",
      "source_content_type_list_attr": "fountain_source_content_filter_ids",
    },
    {
      "name": "source_aid",
      "enable": "{{fountain_enable_source_aid_filter}}",
      "source_aid_attr": "sourcePidAuthorId",
    },
    {
      "name": "follow_author",
      "enable": True,
      "follow_author_filter_timegap_attr": "fountain_follow_author_filter_timegap",
      "author_id_attr": "author__id",
      "upload_time_attr": "upload_time",
    },
    {
      "name": "low_fans_lite",
      "enable": "{{fountain_enable_low_fans_lite_filter_splash}}",
      "count_threshold_attr": "fountain_author_fans_low_bound_splash",
    },
    {
      "name": "low_server_show_lite",
      "enable": "{{fountain_enable_low_server_show_lite_filter_splash}}",
      "count_threshold_attr": "fountain_show_cnt_low_bound_splash",
    },
  ]

  def __init__(self):
    LeafFlow.__init__(self, "filter_splash_v2")
    self \
      .namespace_(ns = "filter_splash_v2", nest = True) \
      ._timestamp_begin("filter_splash") \
      ._filter() \
      ._timestamp_end("filter_splash") \
      ._count_stage_cpu_cost("filter_splash") \
      ._count_photo_type_distribution("filter_finish_splash") \
      ._return_for_fountain_possible() \
      .namespace_()

  def _filter(self):
    return self \
      ._get_commmo_abtest_params() \
      .get_abtest_params(
        biz_name = "RECO_RPC",
        prioritized_suffix = "{{_ABTEST_SUFFIX_LIST_}}",
        ab_params = enrich_ab_param([
        {
            "param_name": "fountain_skip_retrieval_perf_upload_day",
            "param_type": "int",
            "default_value": 1
        },
        {
          "param_name": "fountain_show_cnt_low_bound_splash",
          "param_type": "int",
          "default_value": 500,
        },
        {
          "param_name": "fountain_author_fans_low_bound_splash",
          "param_type": "int",
          "default_value": 200,
        },
        {
          "param_name": "enable_filter_fountain_invalid_hetu_id",
          "param_type": "int",
          "default_value": 0,
        },
        {
          "param_name": "skip_fountain_match_hetu_level_one",
          "param_type": "int",
          "default_value": 1,
        },
        {
          "param_name": "enable_cal_information_score_init",
          "param_type": "int",
          "default_value": 0,
        },
        {
          "param_name": "fountain_skip_filter_photo_by_not_related_information_splash",
          "param_type": "int",
          "default_value": 1,
        },
        {
          "param_name": "fountain_skip_filter_photo_by_not_related_reason_splash_v2",
          "param_type": "int",
          "default_value": 1,
        },
        {
          "param_name": "fountain_splash_use_emb_similarity_score_filter",
          "param_type": "int",
          "default_value": 0,
        },
        {
          "param_name": "fountain_splash_emb_similarity_score_threshold",
          "param_type": "double",
          "default_value": 0.36,
        },
        {
          "param_name": "fountain_skip_empty_source_hetu_filter",
          "param_type": "int",
          "default_value": 1,
        },
        {
          "param_name": "skip_fountain_filter_backup_retr_splash",
          "param_type": "int",
          "default_value": 1,
        },
        {
          "param_name": "fountain_splash_backup_retr_num_threshold",
          "param_type": "int",
          "default_value": 100,
        },
        {
          "param_name": "filter_not_high_quality_out_of_date",
          "param_type": "int",
          "default_value": 0,
        },
        {
          "param_name": "filter_not_high_quality_days",
          "param_type": "int",
          "default_value": 90,
        },
        {
          "param_name": "fountain_skip_trans_hetu_tag_item_attr_new",
          "param_type": "int",
          "default_value": 1,
        },
        {
          "param_name": "fountain_related_score_v2_splash_thres",
          "param_type": "int",
          "default_value": 0,
        },
        ("fountain_enable_splash_retr_filter_limit", False),
        ("fountain_related_score_filter_cancel_num", -1),
        ("fountain_enable_low_server_show_lite_filter_splash", True),
        ("enable_explore_fountain_leaf_use_hetu_v3", False),
        ("enable_fountain_use_hetu_v1_related_score_calc_v2_splash", False),
        ("enable_fountain_dynamic_xtr_filter_splash", False, "enable_fountain_dynamic_xtr_filter"),
        ("fountain_dynamic_xtr_filter_threshold_splash_str", "", "fountain_dynamic_xtr_filter_threshold_str"),
        ("fountain_filter_old_photo_days_splash", 0, "fountain_filter_old_photo_days"),
        ("enable_fountain_save_follow_author_splash", 0, "enable_fountain_save_follow_author"),
        ("fountain_enable_user_reco_neg_photo_filter_splash", False, "fountain_enable_user_reco_neg_photo_filter"),
        ("fountain_enable_source_content_filter_splash", False, "fountain_enable_source_content_filter"),
        ("fountain_source_content_filter_ids_str_splash", "3,8,10,15,16,17", "fountain_source_content_filter_ids_str"),
        ("fountain_enable_source_aid_filter_splash", True, "fountain_enable_source_aid_filter"),
        ("enable_fountain_related_score_calc_v2_use_cluster_id", 0),
        ("enable_fountain_related_score_calc_v2_use_author", 1),
        ("enable_fountain_related_score_calc_v2_use_style_tag", 1),
        ("enable_fountain_related_score_calc_use_hash_tag", False),
        ("enable_fountain_related_score_calc_use_author_circle_v2", False),
        ("enable_fountain_related_score_calc_v3", False),
        ("fountain_splash_emb_similarity_calculate_item_limit", 800),
        ("enable_fountain_splash_extract_related_hetu_tag_v2_ids", False), # 是否抽取相关的 Source 标签，过滤无意义、不相关标签
        ("enable_fountain_splash_extract_related_hetu_tag_v2_ids_use_range_filter", False), # 是否使用标签 range 过滤
        ("fountain_splash_hetu_tag_v2_format_range_lower_bound", 50200), # 形式标签下限
        ("fountain_splash_hetu_tag_v2_format_range_upper_bound", 50300), # 形式标签上限
        ("fountain_splash_hetu_tag_v2_element_range_lower_bound", 55000), # 元素标签下限
        ("fountain_splash_hetu_tag_v2_element_range_upper_bound", 60000), # 元素标签上限
        ("fountain_splash_hetu_tag_v2_content_range_lower_bound", 500006), # 内容标签下限
        ("fountain_splash_hetu_tag_v2_content_range_upper_bound", 4000000), # 内容标签上限
        ("enable_fountain_splash_emb_similarity_score_adjust_by_hetu1", False),
        ("enable_fountain_splash_emb_similarity_score_adjust_by_user_explore_activity", False),
        ("fountain_splash_explore_5min_user_threshold", 0.2),
        ("fountain_splash_emb_similarity_score_adjust_lt_5min_user_coeff", 1.0),
        ("enable_fountain_splash_emb_threshold_adjust_by_user_preference_weight", False),
        ("fountain_splash_related_filter_use_hetu_embedding_v4", False),
        ("fountain_splash_emb_similarity_score_hetu1_threshold_kconf_path", "emb_similarity_score_threshold"),
        ("fountain_splash_related_filter_embedding_dim", 64),
      ])) \
      .split_string(
        input_common_attr = "fountain_source_content_filter_ids_str",
        output_common_attr = "fountain_source_content_filter_ids",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_int = True,
      ) \
      .get_abtest_params(
        biz_name = "RECO_RPC",
        ab_params = [
          {
            "attr_name": "fountain_enable_low_fans_lite_filter_splash",
            "default_value": True,
            "param_name": "fountain_enable_low_fans_lite_filter_splash",
            "param_type": "bool",
            "report_ab_hit": "{{_ABTEST_REPORT_HIT_}}"
          },
        ],
        prioritized_suffix = "{{_ABTEST_SUFFIX_LIST_}}"
      ) \
      .enrich_attr_by_lua(
        import_common_attr = [
          "sourcePidAuthorId",
          "source_hetu_level_one_v2",
          "enable_cal_information_score_init", 
          "enable_explore_fountain_leaf_use_hetu_v3"
        ],
        export_common_attr = [
          "sourcePidAuthorId",
          "enable_cal_information_score_splash"
        ],
        function_for_common = "calculate",
        lua_script_file = "fountain/filter/lua/filter_splash_control.lua") \
       .if_("enable_cal_information_score_splash == 1") \
        .get_kconf_params(
          kconf_configs=[{
          "kconf_key": "reco.fountain.informationHetuTagId",
          "value_type": "list_int64",
          "default_value": [],
          "export_common_attr": "information_hetu_tag_id"
          }]
        ) \
      .end_if_() \
      ._common_filter() \
      .if_("fountain_enable_splash_retr_filter_limit == 1", to_be_delete = "date=2024-05-29;committer=denghong") \
        .explore_retrieval_filter(
          user_info_ptr_attr = "userInfoPb",
          item_attr_map = {**FilterBaseFlow._ITEM_ATTR_MAP, **FilterSplashV2Flow._ITEM_ATTR_MAP},
          filters = FilterBaseFlow._FILTERS + FilterSplashV2Flow._FILTERS,
        ) \
      .else_() \
        .explore_retrieval_filter(
          user_info_ptr_attr = "userInfoPb",
          item_attr_map = {**FilterBaseFlow._ITEM_ATTR_MAP, **FilterSplashV2Flow._ITEM_ATTR_MAP},
          filters = FilterBaseFlow._FILTERS + FilterSplashV2Flow._FILTERS,
          truncation_map = {
            "default": 5000,
          },
        ) \
      .end_() \
      .get_kconf_params(
        kconf_configs = [
          {
            "kconf_key": "reco.index.enableTempratureFobidTimeFilter",
            "value_type": "bool",
            "default_value": False,
            "export_common_attr": "enable_temp_upload_time_filter"
          },
          {
            "kconf_key": "reco.index.tempratureFobidTimeStart",
            "value_type": "int",
            "default_value": 0,
            "export_common_attr": "temp_upload_time_filter_start"
          },
          {
            "kconf_key": "reco.index.tempratureFobidTimeEnd",
            "value_type": "int",
            "default_value": 0,
            "export_common_attr": "temp_upload_time_filter_end"
          },
        ]
      ) \
      .if_("enable_temp_upload_time_filter == 1") \
        .filter_by_kconf_list(
          enable_white = True,
          enable_black = False,
          kconf_key = "reco.grpr.antiHackLiveStreamConfig",
          white_list_name = "photo_aid_white_list_kconf_name_list",
          filter_item_attr = "author__id",
          select_item = {
            "join": "and",
            "filters": [{
              "attr_name": "upload_time",
              "select_if": ">",
              "compare_to": "{{temp_upload_time_filter_start}}",
            }, {
              "attr_name": "upload_time",
              "select_if": "<=",
              "compare_to": "{{temp_upload_time_filter_end}}",
            }]
          }
        ) \
      .end_() \
      .enrich_attr_by_lua(
        import_common_attr = [
          "source_hetu_level_one_v2",
          "source_hetu_level_two_v2",
          "source_hetu_level_three_v2",
          "source_hetu_level_four_v2",
          "sourcePidFourthLevelCategory",
          "sourcePidThirdLevelCategory",
          "source_hetu_face_id_v2",
          "source_hetu_tag_v2",
          "source_hetu_cluster_id_v2",
          "enable_fountain_related_score_calc_v3",
          "fountain_skip_filter_photo_by_not_related_reason_splash_v2",
          "fountain_skip_filter_photo_by_not_related_information_splash"
        ],
        export_common_attr = [
          "enable_fountain_related_score_calc_v3",
          "fountain_skip_filter_photo_by_not_related_reason_splash_v2",
          "fountain_skip_filter_photo_by_not_related_information_splash"
        ],
        function_for_common = "calculate",
        lua_script_file = "fountain/filter/lua/skip_empty_source_hetu.lua",
        skip = "{{fountain_skip_empty_source_hetu_filter}}") \
      .count_reco_result(save_count_to = "fountain_splash_item_num_before_related_filter") \
      .if_("enable_fountain_related_score_calc_v3 == 1") \
        .get_kconf_params(
          kconf_configs = [
            {
              "kconf_key": "reco.fountain.relatedScoreConfig",
              "json_path": "hetu_conf_key",
              "default_value": [-1],
              "export_common_attr": "splash_related_score_v3_hetu_conf_key_list"
            },
            {
              "kconf_key": "reco.fountain.relatedScoreConfig",
              "json_path": "hetu_conf_value",
              "default_value": [2,3,4,5],
              "export_common_attr": "splash_related_score_v3_hetu_conf_value_list"
            },
            {
              "kconf_key": "reco.fountain.relatedScoreConfig",
              "json_path": "tag_element_conf",
              "default_value": 4,
              "export_common_attr": "splash_related_score_v3_tag_element_score"
            },
            {
              "kconf_key": "reco.fountain.relatedScoreConfig",
              "json_path": "tag_content_conf",
              "default_value": 3,
              "export_common_attr": "splash_related_score_v3_tag_content_score"
            },
            {
              "kconf_key": "reco.fountain.relatedScoreConfig",
              "json_path": "user_hash_tag_conf",
              "default_value": 1,
              "export_common_attr": "splash_related_score_v3_user_hash_tag_score"
            },
            {
              "kconf_key": "reco.fountain.relatedScoreConfig",
              "json_path": "ip_conf",
              "default_value": 6,
              "export_common_attr": "splash_related_score_v3_ip_score"
            },
            {
              "kconf_key": "reco.fountain.relatedScoreConfig",
              "json_path": "cid_conf",
              "default_value": 6,
              "export_common_attr": "splash_related_score_v3_cid_score"
            },
            {
              "kconf_key": "reco.fountain.relatedScoreConfig",
              "json_path": "aid_conf",
              "default_value": 2,
              "export_common_attr": "splash_related_score_v3_aid_score"
            },
            {
              "kconf_key": "reco.fountain.relatedScoreConfig",
              "json_path": "author_circle_v2_conf",
              "default_value": 2,
              "export_common_attr": "splash_related_score_v3_author_circle_v2_score"
            },
            {
              "kconf_key": "reco.fountain.relatedScoreConfig",
              "json_path": "filterd_tag_ids_conf",
              "default_value": [58515,58469,58590,58871,59509,59521,59530,58606,59510,58410,58833,59000,
                                59098,59262,58367,57001,56002,58793,58027,58366,56001,59900,58838,3177743,
                                3182042,792812,3193109,3049566],
              "export_common_attr": "splash_filterd_tag_ids"
            },
          ]
        ) \
        .if_("enable_fountain_splash_extract_related_hetu_tag_v2_ids == 1") \
          .enrich_attr_by_light_function(
            import_common_attr = [
              {"name": "source_hetu_tag_v2", "as": "hetu_tag_ids"},
              {"name": "splash_filterd_tag_ids", "as": "balck_hetu_tag_ids"},
              {"name": "enable_fountain_splash_extract_related_hetu_tag_v2_ids_use_range_filter", "as": "use_range_filter"},
              {"name": "fountain_splash_hetu_tag_v2_format_range_lower_bound", "as": "hetu_tag_format_range_lower_bound"},
              {"name": "fountain_splash_hetu_tag_v2_format_range_upper_bound", "as": "hetu_tag_format_range_upper_bound"},
              {"name": "fountain_splash_hetu_tag_v2_element_range_lower_bound", "as": "hetu_tag_element_range_lower_bound"},
              {"name": "fountain_splash_hetu_tag_v2_element_range_upper_bound", "as": "hetu_tag_element_range_upper_bound"},
              {"name": "fountain_splash_hetu_tag_v2_content_range_lower_bound", "as": "hetu_tag_content_range_lower_bound"},
              {"name": "fountain_splash_hetu_tag_v2_content_range_upper_bound", "as": "hetu_tag_content_range_upper_bound"},
            ],
            export_common_attr = [
              "related_source_hetu_tag_v2",
            ],
            function_name = "ExtractRelatedSourceHetuV2Tag",
            class_name = "ExploreLightFunctionSetV2",
          ) \
        .else_() \
          .copy_attr(
            attrs = [
              {
                "from_common": "source_hetu_tag_v2",
                "to_common": "related_source_hetu_tag_v2"
              }
            ]
          ) \
        .end_() \
        .explore_related_score_enricher_v2(
          source_hetu_attr_list = ["source_hetu_level_one_v2", "source_hetu_level_two_v2", "source_hetu_level_three_v2", "source_hetu_level_four_v2"],
          source_author_str_list = ["sourcePidFourthLevelCategory", "sourcePidThirdLevelCategory"],
          source_face_id_attr = "source_hetu_face_id_v2",
          source_hetu_tag_attr = "related_source_hetu_tag_v2",
          source_cluster_id_attr = "source_hetu_cluster_id_v2",
          source_aid_attr = "sourcePidAuthorId",
          source_hetu_cid_attr = "source_hetu_sim_cluster_id",
          source_user_hash_tag_id_attr = "source_user_hash_tag_id",
          source_author_circle_v2_attr = "source_author_circle_v2",
          target_hetu_attr_list = ["hetu_tag_level_info_v2__hetu_level_one", "hetu_tag_level_info_v2__hetu_level_two", "hetu_tag_level_info_v2__hetu_level_three", "hetu_tag_level_info_v2__hetu_level_four"],
          target_author_str_list = ["author__category_detail__fourth_level_id", "author__category_detail__third_level_id"],
          target_face_id_attr = "hetu_tag_level_info_v2__hetu_face_id",
          target_hetu_tag_attr = "hetu_tag_level_info_v2__hetu_tag",
          target_cluster_id_attr = "hetu_tag_level_info_v2__hetu_cluster_id",
          target_aid_attr = "author__id",
          target_hetu_cid_attr = "hetu_sim_cluster_id",
          target_user_hash_tag_id_attr = "user_hash_tag_id",
          target_author_circle_v2_attr = "author_circle_v2",
          save_score_to_attr = "fountain_related_score_v2",
          save_score_detail_to_attr = "fountain_related_score_v2_detail",
          hetu_conf_score_key_list = "splash_related_score_v3_hetu_conf_key_list",
          hetu_conf_score_value_list = "splash_related_score_v3_hetu_conf_value_list",
          tag_element_conf_score = "splash_related_score_v3_tag_element_score",
          tag_content_conf_score = "splash_related_score_v3_tag_content_score",
          user_hash_tag_conf_score = "splash_related_score_v3_user_hash_tag_score",
          ip_conf_score = "splash_related_score_v3_ip_score",
          cid_conf_score = "splash_related_score_v3_cid_score",
          aid_conf_score = "splash_related_score_v3_aid_score",
          author_circle_v2_conf_score = "splash_related_score_v3_author_circle_v2_score",
          enable_use_author = True,
          enable_v3 = True,
          enable_use_hash_tag = "{{enable_fountain_related_score_calc_use_hash_tag}}",
          enable_use_author_circle_v2 = "{{enable_fountain_related_score_calc_use_author_circle_v2}}",
          target_reason = [
            10002, 10038, 10046, 10071, 10082, 10083, 10084, 10088, 10098, 10135, 10143, 10147,
            10149, 10150, 10300, 10301, 10308, 10310, 10311, 10312, 10317, 10318, 10324, 10325,
            10326, 10328, 10329, 10400, 10401, 10402, 10403, 10405, 10406, 10407, 10408, 10424, 10302,
            10426, 10788, 10790, 11207, 11208, 11501, 11502, 10136, 10417, 13020, 13021, 10303,
            10401, 10151, 10152, 10313, 10314, 13017, 13026, 10409, 10414, 10415, 10239, 10461, 10462,
            10411, 10412, 10413, 10416, 10418, 10010, 10384
          ]
        ) \
      .else_() \
        .if_("enable_fountain_use_hetu_v1_related_score_calc_v2_splash == 1") \
          .explore_related_score_enricher_v2(
            source_hetu_attr_list = ["source_hetu_level_one", "source_hetu_level_two", "source_hetu_level_three", "source_hetu_level_four"],
            source_author_str_list = ["sourcePidFourthLevelCategory", "sourcePidThirdLevelCategory"],
            source_face_id_attr = "source_hetu_face_ids",
            source_hetu_tag_attr = "source_hetu_tag_level_info_hetu_tag",
            source_cluster_id_attr = "source_hetu_cluster_ids",
            target_hetu_attr_list = ["hetu_tag_level_info__hetu_level_one", "hetu_tag_level_info__hetu_level_two", "hetu_tag_level_info__hetu_level_three", "hetu_tag_level_info__hetu_level_four"],
            target_author_str_list = ["author__category_detail__fourth_level_id", "author__category_detail__third_level_id"],
            target_face_id_attr = "hetu_tag_level_info__hetu_face_id",
            target_hetu_tag_attr = "hetu_tag_level_info__hetu_tag",
            target_cluster_id_attr = "hetu_tag_level_info__hetu_cluster_id",
            save_score_to_attr = "fountain_related_score_v2",
            save_score_detail_to_attr = "fountain_related_score_v2_detail",
            enable_hetu_v1 = True,
            enable_use_cluster_id = "{{enable_fountain_related_score_calc_v2_use_cluster_id}}",
            enable_use_author = "{{enable_fountain_related_score_calc_v2_use_author}}",
            enable_use_style_tag = "{{enable_fountain_related_score_calc_v2_use_style_tag}}",
            target_reason = [
              10002, 10038, 10046, 10071, 10082, 10083, 10084, 10088, 10098, 10135, 10143, 10147,
              10149, 10150, 10300, 10301, 10308, 10310, 10311, 10312, 10317, 10318, 10324, 10325,
              10326, 10328, 10329, 10400, 10401, 10402, 10403, 10405, 10406, 10407, 10408, 10424, 10302,
              10426, 10788, 10790, 11207, 11208, 11501, 11502, 10136, 10417, 13020, 13021, 10303,
              10401, 10151, 10152, 10313, 10314, 13017, 13026, 10409, 10414, 10415, 10239, 10461, 10462,
              10411, 10412, 10413
            ]
          ) \
        .end_() \
        .if_("enable_cal_information_score_splash == 1") \
          .explore_information_related_score_enricher(
            information_related_score = "information_related_score",
            source_hetu_attr_list = ["source_hetu_level_three_v2", "source_hetu_level_two_v2"],
            source_author_str_list = ["sourcePidFourthLevelCategory", "sourcePidThirdLevelCategory"],
            source_face_id_attr = "source_hetu_face_id_v2",
            source_hetu_tag_attr = "source_hetu_tag_v2",
            source_cluster_id_attr = "source_hetu_cluster_id_v2",
            target_hetu_attr_list = ["hetu_tag_level_info_v2__hetu_level_three", "hetu_tag_level_info_v2__hetu_level_two"],
            target_author_str_list = ["author__category_detail__fourth_level_id", "author__category_detail__third_level_id"],
            target_face_id_attr = "hetu_tag_level_info_v2__hetu_face_id",
            target_hetu_tag_attr = "hetu_tag_level_info_v2__hetu_tag",
            target_cluster_id_attr = "hetu_tag_level_info_v2__hetu_cluster_id",
            information_hetu_tag_id_attr = "information_hetu_tag_id",
            use_hetu_v3 = "{{enable_explore_fountain_leaf_use_hetu_v3}}"
          ) \
        .end_if_() \
      .end_() \
      .if_("fountain_splash_use_emb_similarity_score_filter == 1") \
        .calc_emb_sim_score_and_filter() \
      .else_() \
        .if_("fountain_skip_filter_photo_by_not_related_reason_splash_v2 == 0") \
          ._filter_by_attr_with_perf(
            attr_name = "fountain_related_score_v2",
            remove_if = "<=",
            compare_to = "{{fountain_related_score_v2_splash_thres}}",
            remove_if_attr_missing = False,
            cancel_num = "{{fountain_related_score_filter_cancel_num}}"
          ) \
        .end_() \
      .end_() \
      ._filter_by_attr_with_perf(
        attr_name = "information_related_score",
        remove_if = "==",
        compare_to = 0,
        remove_if_attr_missing = False,
        cancel_num = "{{fountain_related_score_filter_cancel_num}}",
        skip = "{{fountain_skip_filter_photo_by_not_related_information_splash}}") \
      .copy_item_meta_info(
        save_reason_to_attr = "reason") \
      .transform_item_attr(
        mappings = [
          {
            "check_attr_name": "author__id",
            "check_attr_type": "int",
            "output_attr_name": "is_photo_author_followed",
            "output_attr_type": "int",
            # 检查规则
            "rules": [{
              # 当 author__id 在 followAuthors 内时
              "check_values": ["{{followAuthors}}"],
              "output_value": 1,
            }]
          },
        ]
      ) \
      .set_attr_value(
        item_attrs=[
          {
            "name": "is_follow_author", # 关注作者建议使用此字段
            "type": "int",
            "value": 1
          }
        ],
        target_item = {
          "is_photo_author_followed": 1
        },
      ) \
      .count_item_attr(
        counters = [{
          "check_attr_name": "hetu_tag_level_info__hetu_level_one",
          "check_values": [
            "{{source_hetu_level_one}}"
          ],
          "output_attr_name": "is_photo_same_hetu_level_one"
        },
        {
          "check_attr_name": "author__category_detail__third_level_id",
          "check_values": [
            "{{source_author_third_level_id}}"
          ],
          "output_attr_name": "is_photo_same_author_third_level_id"
        },
        {
          "check_attr_name": "hetu_tag_level_info__hetu_level_two",
          "check_values": [
            "{{source_hetu_level_two}}"
          ],
          "output_attr_name": "is_photo_same_hetu_level_two"
        },
        {
          "check_attr_name": "hetu_tag_level_info__hetu_level_three",
          "check_values": [
            "{{source_hetu_level_three}}"
          ],
          "output_attr_name": "is_photo_same_hetu_level_three"
        },
        {
          "check_attr_name": "hetu_tag_level_info__hetu_level_four",
          "check_values": [
            "{{source_hetu_level_four}}"
          ],
          "output_attr_name": "is_photo_same_hetu_level_four"
        },
        {
          "check_attr_name": "hetu_tag_level_info__hetu_tag",
          "check_values": [
            "{{source_hetu_tag_level_info_hetu_tag}}"
          ],
          "output_attr_name": "is_photo_same_hetu_tag"
        },
        {
          "check_attr_name": "hetu_tag_level_info__hetu_face_id",
          "check_values": [
            "{{source_hetu_face_ids}}"
          ],
          "output_attr_name": "is_photo_same_hetu_face_id"
        }]) \
      .if_("fountain_skip_trans_hetu_tag_item_attr_new == 0") \
        .enrich_attr_by_lua(
          import_item_attr = [
            "hetu_tag_level_info__hetu_level_one",
            "hetu_tag_level_info__hetu_level_two",
          ],
          export_item_attr = [
            "hetu_level_one",
            "hetu_level_two",
          ],
          function_for_item = "calculate",
          lua_script_file = "fountain/filter/lua/trans_hetu_tagv2.lua") \
        .explore_transform_hetu_tag(
          output_attrs = ["hetu_level_one_v2", "hetu_level_two_v2",  "hetu_level_three_v2", "hetu_level_four_v2", "hetu_tag_v2", "hetu_face_id_v2"],
          hetu_tag_attrs = ["hetu_tag_level_info_v2__hetu_level_one", "hetu_tag_level_info_v2__hetu_level_two", "hetu_tag_level_info_v2__hetu_level_three", "hetu_tag_level_info_v2__hetu_level_four", "hetu_tag_level_info_v2__hetu_tag", "hetu_tag_level_info_v2__hetu_face_id"]
        ) \
      .end_if_() \
      .count_reco_result(save_count_to = "fountain_splash_item_num_after_filter") \
      .if_("fountain_splash_item_num_before_related_filter > fountain_splash_item_num_after_filter") \
        .set_attr_value(
          common_attrs=[
            {
              "name": "filter_splash_related_score_filtered_success", # 统计当前请求是否成功启用相关过滤
              "type": "int",
              "value": 1
            }
          ]
        ) \
      .end_() \
      .count_reco_result(
        save_count_to = "backup_retr_item_num",
        target_reason = [10406] # 兜底召回源
      ) \
      .if_("skip_fountain_filter_backup_retr_splash == 0 and fountain_splash_item_num_after_filter - backup_retr_item_num >= fountain_splash_backup_retr_num_threshold", to_be_delete = "date=2024-05-29;committer=denghong") \
        ._filter_by_attr_with_perf(
          attr_name = "reason",
          remove_if = "==",
          compare_to = 10406,
          remove_if_attr_missing = False
        )\
      .end_if_() \
      .log_debug_info(
        item_attrs = [
          "hetu_level_one_v2",
          "hetu_level_two_v2",
          "hetu_level_three_v2",
          "hetu_level_four_v2",
          "hetu_tag_v2",
          "hetu_face_id_v2",
          "hetu_level_one",
          "hetu_level_two",
          "fountain_related_score_v2",
          "is_follow_author"
        ],
        common_attrs = [
          "source_hetu_level_one_v2",
          "source_hetu_level_two_v2",
          "source_hetu_level_three_v2",
          "source_hetu_level_four_v2",
          "featureUserRequestCityId",
          "sourceIsLocalLifePhoto",
          "mock_cb2cf_user_emb",
        ],
        item_num_limit = 10,
        for_debug_request_only = True,
      ) \
      .fountain_environment_perf_log(
        skip = "{{fountain_skip_retrieval_perf_upload_day}}",
        enable_upload_day_perf = True,
        upload_time_attr = "upload_time",
        upload_day_divide = "0-1-2-3-4-5-6-30-60-120-180",
        check_point = "fountain.retrieval"
        ) \
      .perflog_attr_value(
        check_point = "fountain.retrieval",
        item_attrs = [
          "duration_ms",
        ]) \
      .perflog_reason_count(
        name = "fountain_splash_retr_filter",
        traceback = True,
        check_point = "filter_finish"
      ) \
      .count_reco_result(
        save_count_to = "filter_finish_splash_item_num"
      )

  # 这段代码是希望在原有规则相似的基础上, 增加 embedding 相似的判定, 以便捞回一些规则字段不完整导致的对相似召回结果的误伤
  # 使用 mmu 产出的基于内容的 embedding server, 对会被规则过滤的 pids, 计算它们与入口的相似度, 将高相似的 pid 捞回
  def calc_emb_sim_score_and_filter(self):
    return self \
      .adjust_emb_sim_score_threshold() \
      .pack_item_attr(
        target_item = {"fountain_related_score_v2": 0},
        item_source = {
          "reco_results": True,
        },
        mappings = [{
          "from_item_attr": "photo_id",
          "to_common_attr": "zero_related_score_photos",
          "item_attr_limit": "{{fountain_splash_emb_similarity_calculate_item_limit}}"
        }]
      ) \
      .pack_common_attr(
        input_common_attrs = ["zero_related_score_photos", "featureSourcePId"],
        output_common_attr = "embedding_source_pids",
      ) \
      .if_("fountain_splash_related_filter_use_hetu_embedding_v4 == 1") \
        .fetch_remote_embedding(
          protocol = 1,
          shard_num = 8,
          timeout_ms = 10,
          colossusdb_embd_model_name = "explore_reco_hetu_emb_v4",
          colossusdb_embd_table_name = "explore_reco_hetu_emb_v4",
          id_converter = {"type_name": "mioEmbeddingIdConverter"},
          input_attr_name = "embedding_source_pids",
          output_attr_name = "mmu_embeddings",
          query_source_type = "common_attr",
          size = 128,
          client_side_shard = True,
        ) \
      .else_() \
        .get_remote_embedding_lite(
          kess_service = "grpc_MMUHetuContentEmbeddingV2",
          shard_num = 4,
          timeout_ms = 10,
          id_converter = {"type_name": "kuibaEmbeddingIdConverter"},
          input_attr_name = "embedding_source_pids",
          output_attr_name = "mmu_embeddings",
          query_source_type = "common_attr",
          size = 64,
          client_side_shard = True
        ) \
      .end_() \
      .explore_custom_embedding_score_enricher(
        target_item = {"fountain_related_score_v2": 0},
        enable_fix_low_hit_rate = True,
        user_info_ptr_attr = "userInfoPb",
        embedding_list_attr = "mmu_embeddings",
        source_pids_list_attr = "embedding_source_pids",
        calc_type = "single_dot",
        export_item_attr = "splash_retr_similary_score",
        dim_size = "{{fountain_splash_related_filter_embedding_dim}}",
        source_photo_id = "{{featureSourcePId}}"
      ) \
      .set_attr_value( # 给一个默认值让规则过滤掉, 主要针对emb获取失败的photo, 没有这个分数 
        no_overwrite = True,
        item_attrs = [
          {
            "name": "emb_similary_photo",
            "type": "int",
            "value": 0
          }
        ]
      ) \
      .transform_item_attr(
        mappings = [
          {
            "check_attr_name": "splash_retr_similary_score",
            "check_attr_type": "double",
            "output_attr_name": "emb_similary_photo",
            "output_attr_type": "int",
            "output_default_value": 0,
            "rules": [
              {
                "check_range": {
                  "lower_bound": "{{fountain_splash_emb_similarity_score_threshold}}", # 包含，可缺省
                },
                "output_value": 1
              },
            ]
          }
        ]
      ) \
      ._filter_by_attr_with_perf(
        target_item = {"emb_similary_photo": 0},
        attr_name = "fountain_related_score_v2",
        remove_if = "<=",
        compare_to = "{{fountain_related_score_v2_splash_thres}}",
        remove_if_attr_missing = False,
        cancel_num = "{{fountain_related_score_filter_cancel_num}}"
      )

  def adjust_emb_sim_score_threshold(self):
    return self \
      .if_("enable_fountain_splash_emb_similarity_score_adjust_by_hetu1 == 1") \
        .get_kconf_params(  # 分类目阈值
          kconf_configs = [
            {
              "kconf_key": "reco.fountain.relatedScoreConfig",
              "json_path": "{{fountain_splash_emb_similarity_score_hetu1_threshold_kconf_path}}",
              "default_value": [0.3,0.4,0.3,0.35,0.35,0.45,0.4,0.3,0.3,0.7,0.3,0.4,0.46,0.3,0.35,0.4,0.3,0.4,0.35,0.3,0.35,0.4,0.3,0.35,0.35,0.35,0.4,0.4,0.35,0.3,0.4,0.35,0.3,0.35,0.4,0.3,0.4,0.4,0.35],
              "export_common_attr": "emb_similarity_score_threshold_list"
            },
          ]
        ) \
        .gen_common_attr_by_lua(
          attr_map={
            "source_hetu_level_one_v2_first_index": "(source_hetu_level_one_v2 and #source_hetu_level_one_v2 > 0) and (source_hetu_level_one_v2[1] - 1) or -1",
          }
        ) \
        .select_list_values(
          index_attr = "source_hetu_level_one_v2_first_index",
          list_values = [
            {"from": "emb_similarity_score_threshold_list", "to": "fountain_splash_emb_similarity_score_threshold"},
          ],
          is_common_attr=True
        ) \
      .end_() \
      .if_("enable_fountain_splash_emb_similarity_score_adjust_by_user_explore_activity == 1 and active_days_gt_5min_rate < fountain_splash_explore_5min_user_threshold") \
        .gen_common_attr_by_lua(  # 5 min+ 用户调整
          attr_map={
            "fountain_splash_emb_similarity_score_threshold": "fountain_splash_emb_similarity_score_threshold * fountain_splash_emb_similarity_score_adjust_lt_5min_user_coeff",
          }
        ) \
      .end_() \
      .if_("enable_fountain_splash_cacl_long_term_relevance_preference_weight == 1 and enable_fountain_splash_emb_threshold_adjust_by_user_preference_weight == 1") \
        .gen_common_attr_by_lua(  # 根据大盘相关偏好调整阈值
          attr_map={
            "fountain_splash_emb_similarity_score_threshold": "fountain_splash_emb_similarity_score_threshold * user_long_term_relevance_preference_weight",
          }
        ) \
      .end_()

  def _return_for_fountain_possible(self):
    return self \
      .if_("is_fountain_possible == 0") \
        .sort(
          score_from_attr = "fountain_related_score_v2",
        ) \
        .limit(30) \
        .enrich_attr_by_lua(
          import_item_attr = [
            "fountain_stats__real_show_count",
            "fountain_stats__like_count",
            "fountain_stats__forward_count",
            "fountain_stats__follow_count",
            "fountain_stats__negative_count",
            "fountain_stats__view_length_sum",
          ],
          export_item_attr = [
            "fullrank_detail_pltr",
            "fullrank_detail_pftr",
            "fullrank_detail_pwtr",
            "fullrank_detail_phtr",
            "fullrank_sim_pfintr",
          ],
          function_for_item = "calculate",
          lua_script = """
            function calculate(seq, item_key, reason, score)
              local total_count = fountain_stats__real_show_count or 0
              if total_count <= 0 then
                return 0.0, 0.0, 0.0, 0.0, 0.0
              end
              local like_count = fountain_stats__like_count or 0
              local forward_count = fountain_stats__forward_count or 0
              local follow_count = fountain_stats__follow_count or 0
              local negative_count = fountain_stats__negative_count or 0
              local view_length_sum = fountain_stats__view_length_sum or 0
              return like_count * 1.0 / total_count, forward_count * 1.0 / total_count,
                follow_count * 1.0 / total_count, negative_count * 1.0 / total_count,
                view_length_sum * 0.001 / total_count
            end
          """
        ) \
        .return_() \
      .end_()
