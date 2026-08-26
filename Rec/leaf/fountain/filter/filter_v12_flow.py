#!/usr/bin/env python3
# coding=utf-8

from dragonfly.common_leaf_dsl import LeafFlow
from dragonfly.ext.explore.explore_api_mixin import ExploreApiMixin
from dragonfly.ext.subdivision.subdivision_api_mixin import subdivisionApiMixin
from filter.filter_base_flow import FilterBaseFlow
from util import enrich_ab_param


class FilterV12Flow(FilterBaseFlow, subdivisionApiMixin, ExploreApiMixin):
  _ITEM_ATTR_MAP = {
  }

  _FILTERS = [
    {
      "name": "follow_author",
      "enable": "{{fountain_enable_follow_author_filter}}",
      "follow_author_filter_timegap_attr": "fountain_follow_author_filter_timegap",
      "author_id_attr": "author__id",
      "upload_time_attr": "upload_time",
    },
  ]

  def __init__(self):
    LeafFlow.__init__(self, "filter_v12")
    self \
      .namespace_(ns = "filter_12", nest = True) \
      ._timestamp_begin("filter_fast") \
      ._filter() \
      ._timestamp_end("filter_fast") \
      ._count_stage_cpu_cost("filter_fast") \
      ._count_photo_type_distribution("filter_finish_fast") \
      .namespace_()

  def _filter(self):
    return self \
      ._get_commmo_abtest_params() \
      .get_abtest_params(
        biz_name = "RECO_RPC",
        prioritized_suffix = "{{_ABTEST_SUFFIX_LIST_}}",
        ab_params = enrich_ab_param([
        {
            "param_name": "fountain_hot_high_photo_skip_filter_types_str",
            "param_type": "string",
            "default_value": "9,8,10,14,15,3,17,16"
        },  
        {
            "param_name": "fountain_enable_skip_filter_hot_high_photo_in_bs_fast_v1",
            "param_type": "bool",
            "default_value": False
        },
        {
            "param_name": "fountain_enable_skip_filter_hot_high_photo_in_pagesize",
            "param_type": "bool",
            "default_value": False
        },
        {
            "param_name": "fountain_skip_dedup_content_ids_in_pagesize",
            "param_type": "int",
            "default_value": 1
        },
        {
            "param_name": "fountain_skip_retrieval_perf_upload_day",
            "param_type": "int",
            "default_value": 1
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
          "param_name": "fountain_mmu_content_filter_item_type",
          "param_type": "int",
          "default_value": -1,
        },
        {
          "param_name": "fountain_mmu_content_filter_use_old_id",
          "param_type": "bool",
          "default_value": True,
        },
        ("fountain_need_filter_content_types", "3,8,9"),
        ("fountain_cid_browse_set_hetu_list_str", ""),
        ("fountain_enable_fast_retr_filter_limit", False),
        ("fountain_enable_follow_author_filter", False),
        ("skip_fountain_dedup_content_ids_shuffle", 1),
        ("enable_fountain_dynamic_xtr_filter_fast", False, "enable_fountain_dynamic_xtr_filter"),
        ("fountain_dynamic_xtr_filter_threshold_fast_str", "", "fountain_dynamic_xtr_filter_threshold_str"),
        ("fountain_filter_old_photo_days_fast", 0, "fountain_filter_old_photo_days"),
        ("enable_fountain_save_follow_author_fast", 0, "enable_fountain_save_follow_author"),
        ("fountain_enable_user_reco_neg_photo_filter_fast", False, "fountain_enable_user_reco_neg_photo_filter"),
        ("fountain_enable_fetch_rank_neg_photo_fast", False, "fountain_enable_fetch_rank_neg_photo"),
        ("fountain_enable_fetch_rerank_neg_photo_fast", False, "fountain_enable_fetch_rerank_neg_photo"),
        ("fountain_enable_fetch_mc_neg_photo_fast", False, "fountain_enable_fetch_mc_neg_photo"),
        ("enable_fountain_calc_source_thresh_score_fast", 0),
      ])) \
      ._common_filter() \
      .if_("fountain_enable_fast_retr_filter_limit == 1", to_be_delete = "date=2024-05-29;committer=denghong") \
        .explore_retrieval_filter(
          user_info_ptr_attr = "userInfoPb",
          item_attr_map = {**FilterBaseFlow._ITEM_ATTR_MAP, **FilterV12Flow._ITEM_ATTR_MAP},
          filters = FilterBaseFlow._FILTERS + FilterV12Flow._FILTERS,
        ) \
      .else_() \
        .explore_retrieval_filter(
          user_info_ptr_attr = "userInfoPb",
          item_attr_map = {**FilterBaseFlow._ITEM_ATTR_MAP, **FilterV12Flow._ITEM_ATTR_MAP},
          filters = FilterBaseFlow._FILTERS + FilterV12Flow._FILTERS,
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
      .split_string(
        input_common_attr = "fountain_hot_high_photo_skip_filter_types_str",
        output_common_attr = "fountain_hot_high_photo_skip_filter_types",
        delimiters = ",",
        parse_to_int = True,
        skip = "{{fountain_skip_dedup_content_ids_in_pagesize}}"
      ) \
      .split_string(
        input_common_attr = "fountain_cid_browse_set_hetu_list_str",
        output_common_attr = "cid_browse_set_hetu_list",
        delimiters = ",",
        parse_to_int = True,
      ) \
      .shuffle(
        skip = "{{skip_fountain_dedup_content_ids_shuffle}}"
      ) \
      .fountain_dedup_content_ids_in_pagesize(
        audit_hot_high_tag_level_attr = "audit_hot_high_tag_level",
        hot_high_photo_skip_filter_types = "{{fountain_hot_high_photo_skip_filter_types}}",
        need_filter_types = "{{fountain_need_filter_content_types}}",
        version = 3,
        item_type_of_checked_id = "{{fountain_mmu_content_filter_item_type}}",
        use_old_mmu_content_id = "{{fountain_mmu_content_filter_use_old_id}}",
        mmu_content_attrs = {
          "3": "mmu_content_ids_3",
          "8": "mmu_content_ids_8",
          "9": "mmu_content_ids_9",
          "10": "mmu_content_ids_10",
          "14": "mmu_content_ids_14",
          "15": "mmu_content_ids_15",
          "16": "mmu_content_ids_16",
          "17": "mmu_content_ids_17",
        },
        enable_skip_filter_hot_high_photo = "{{fountain_enable_skip_filter_hot_high_photo_in_pagesize}}",
        enable_skip_filter_hot_high_photo_in_bs = "{{fountain_enable_skip_filter_hot_high_photo_in_bs_fast_v1}}",
        hetu_level_one_attr = "hetu_tag_level_info__hetu_level_one",
        cid_browse_set_hetu_list = "{{cid_browse_set_hetu_list}}",
        skip = "{{fountain_skip_dedup_content_ids_in_pagesize}}") \
      .if_("enable_fountain_calc_source_thresh_score_fast == 1") \
        .calc_source_thresh_score_fast() \
      .end_() \
      .copy_item_meta_info(
        save_reason_to_attr = "reason") \
      .perflog_reason_count(
        skip = "{{fountain_skip_dedup_content_ids_in_pagesize}}",
        check_point = "filter_by_dedup_content_ids_in_pagesize"
      ) \
      .fountain_environment_perf_log(
        skip = "{{fountain_skip_retrieval_perf_upload_day}}",
        enable_upload_day_perf = True,
        upload_time_attr = "upload_time",
        upload_day_divide = "0-1-2-3-4-5-6-30-60-120-180",
        check_point = "fountain.retrieval") \
      .perflog_attr_value(
        check_point = "fountain.retrieval",
        item_attrs = [
          "duration_ms",
        ]) \
      .perflog_reason_count(
        name = "fountain_retr_filter",
        traceback = True,
        check_point = "filter_finish"
      ) \
      .count_reco_result(
        save_count_to = "filter_finish_fast_item_num"
      )

  def calc_source_thresh_score_fast(self):
    return self \
      .get_kconf_params(
        kconf_configs = [
          {
            "kconf_key": "reco.fountain.relatedScoreConfig",
            "json_path": "hetu_conf_key",
            "default_value": [-1],
            "export_common_attr": "related_score_v3_hetu_conf_key_list"
          },
          {
            "kconf_key": "reco.fountain.relatedScoreConfig",
            "json_path": "hetu_conf_value",
            "default_value": [2,3,4,5],
            "export_common_attr": "related_score_v3_hetu_conf_value_list"
          },
          {
            "kconf_key": "reco.fountain.relatedScoreConfig",
            "json_path": "tag_element_conf",
            "default_value": 4,
            "export_common_attr": "related_score_v3_tag_element_score"
          },
          {
            "kconf_key": "reco.fountain.relatedScoreConfig",
            "json_path": "tag_content_conf",
            "default_value": 3,
            "export_common_attr": "related_score_v3_tag_content_score"
          },
          {
            "kconf_key": "reco.fountain.relatedScoreConfig",
            "json_path": "ip_conf",
            "default_value": 6,
            "export_common_attr": "related_score_v3_ip_score"
          },
          {
            "kconf_key": "reco.fountain.relatedScoreConfig",
            "json_path": "cid_conf",
            "default_value": 6,
            "export_common_attr": "related_score_v3_cid_score"
          },
          {
            "kconf_key": "reco.fountain.relatedScoreConfig",
            "json_path": "aid_conf",
            "default_value": 2,
            "export_common_attr": "related_score_v3_aid_score"
          },
        ]
      ) \
      .explore_related_score_enricher_v2(
        source_hetu_attr_list = ["source_hetu_level_one_v2", "source_hetu_level_two_v2", "source_hetu_level_three_v2", "source_hetu_level_four_v2"],
        source_author_str_list = ["sourcePidFourthLevelCategory", "sourcePidThirdLevelCategory"],
        source_face_id_attr = "source_hetu_face_id_v2",
        source_hetu_tag_attr = "source_hetu_tag_v2",
        source_cluster_id_attr = "source_hetu_cluster_id_v2",
        source_aid_attr = "sourcePidAuthorId",
        source_hetu_cid_attr = "source_hetu_sim_cluster_id",
        target_hetu_attr_list = ["hetu_tag_level_info_v2__hetu_level_one", "hetu_tag_level_info_v2__hetu_level_two", "hetu_tag_level_info_v2__hetu_level_three", "hetu_tag_level_info_v2__hetu_level_four"],
        target_author_str_list = ["author__category_detail__fourth_level_id", "author__category_detail__third_level_id"],
        target_face_id_attr = "hetu_tag_level_info_v2__hetu_face_id",
        target_hetu_tag_attr = "hetu_tag_level_info_v2__hetu_tag",
        target_cluster_id_attr = "hetu_tag_level_info_v2__hetu_cluster_id",
        target_aid_attr = "author__id",
        target_hetu_cid_attr = "hetu_sim_cluster_id",
        save_score_to_attr = "fountain_related_score_v2",
        save_score_detail_to_attr = "fountain_related_score_v2_detail",
        hetu_conf_score_key_list = "related_score_v3_hetu_conf_key_list",
        hetu_conf_score_value_list = "related_score_v3_hetu_conf_value_list",
        tag_element_conf_score = "related_score_v3_tag_element_score",
        tag_content_conf_score = "related_score_v3_tag_content_score",
        ip_conf_score = "related_score_v3_ip_score",
        cid_conf_score = "related_score_v3_cid_score",
        aid_conf_score = "related_score_v3_aid_score",
        enable_use_author = True,
        enable_v3 = True,
      )