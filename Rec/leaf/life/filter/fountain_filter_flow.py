from common import CommonRecoFlow
from dragonfly.ext.explore.explore_api_mixin import ExploreApiMixin
from dragonfly.ext.explore_life.explore_life_api_mixin import ExploreLifeApiMixin
from dragonfly.ext.subdivision.subdivision_api_mixin import subdivisionApiMixin
from util import enrich_ab_param


class FountainFilterFlow(CommonRecoFlow, ExploreApiMixin, ExploreLifeApiMixin, subdivisionApiMixin):
  def __init__(self, name: str) -> None:
    super().__init__(name, "life", "filter", "config", "module", "config/module", "lua/module")

  def base_params(self):
    self.namespace_(ns="fountain_filter_flow", nest=True) \
      .get_abtest_params(
        biz_name="RECO_RPC",
        ab_params=enrich_ab_param([
          {
            "attr_name": "fountain_over_days_filter_days_limit",
            "param_name": "fountain_over_days_filter_days_limit",
            "default_value": 180,
            "param_type": "int",
          },
          {
            "attr_name": "fountain_entertainment_hetu_days_limit_attr",
            "param_name": "fountain_entertainment_hetu_days_limit_attr",
            "default_value": 90,
            "param_type": "int",
          },
          {
            "attr_name": "fountain_over_days_filter_low_like_limit",
            "param_name": "fountain_over_days_filter_low_like_limit",
            "default_value": 50,
            "param_type": "int",
          },
          {
            "attr_name": "fountain_over_days_filter_low_like_days_limit",
            "param_name": "fountain_over_days_filter_low_like_days_limit",
            "default_value": 7,
            "param_type": "int",
          },
          {
            "attr_name": "fountain_impression_audit_gray_hours_limit",
            "param_name": "fountain_impression_audit_gray_hours_limit",
            "default_value": 48,
            "param_type": "int",
          },
          {
            "attr_name": "fountain_impression_audit_normal_days_limit",
            "param_name": "fountain_impression_audit_normal_days_limit",
            "default_value": 30,
            "param_type": "int",
          },
          {
            "attr_name": "fountain_impression_audit_high_quality_days_limit",
            "param_name": "fountain_impression_audit_high_quality_days_limit",
            "default_value": 30,
            "param_type": "int",
          },
          {
            "attr_name": "fountain_high_hot_audit_gray_hours_limit",
            "param_name": "fountain_high_hot_audit_gray_hours_limit",
            "default_value": 48,
            "param_type": "int",
          },
          {
            "attr_name": "fountain_high_hot_audit_normal_days_limit",
            "param_name": "fountain_high_hot_audit_normal_days_limit",
            "default_value": 90,
            "param_type": "int",
          },
          {
            "attr_name": "fountain_high_hot_audit_high_quality_days_limit",
            "param_name": "fountain_high_hot_audit_high_quality_days_limit",
            "default_value": 180,
            "param_type": "int",
          },
          {
            "attr_name": "fountain_entertainment_hetu_tag_str",
            "param_name": "fountain_entertainment_hetu_tag_str",
            "default_value": "",
            "param_type": "string",
          },
          {
            "attr_name": "fountain_over_days_filter_topn_screen_map",
            "param_name": "fountain_over_days_filter_topn_screen_map",
            "default_value": "",
            "param_type": "string",
          },
          {
            "attr_name": "enable_fountain_over_days_filter_low_like",
            "param_name": "enable_fountain_over_days_filter_low_like",
            "default_value": False,
            "param_type": "bool",
          },
          {
            "attr_name": "enable_fountain_over_days_filter_audit",
            "param_name": "enable_fountain_over_days_filter_audit",
            "default_value": False,
            "param_type": "bool",
          },
          {
            "attr_name": "fountain_impression_audit_second_level_white_tags",
            "param_name": "enable_fountain_over_days_filter_audit",
            "default_value": "2008292,2008293",
            "param_type": "string",
          },
          {
            "attr_name": "fountain_impression_audit_second_level_black_tags",
            "param_name": "fountain_impression_audit_second_level_black_tags",
            "default_value": "",
            "param_type": "string",
          },
          {
            "attr_name": "fountain_high_hot_audit_second_level_white_tags",
            "param_name": "fountain_high_hot_audit_second_level_white_tags",
            "default_value": "2008226",
            "param_type": "string",
          },
          {
            "attr_name": "fountain_high_hot_audit_second_level_black_tags",
            "param_name": "fountain_high_hot_audit_second_level_black_tags",
            "default_value": "",
            "param_type": "string",
          },
          {
            "attr_name": "fountain_enable_picture_type_filter",
            "default_value": True,
            "param_name": "fountain_enable_picture_type_filter",
            "param_type": "bool",
            "report_ab_hit": "{{_ABTEST_REPORT_HIT_}}"
          },
          {
            "attr_name": "fountain_skip_filter_by_picture_single_variant_attr",
            "default_value": 0,
            "param_name": "fountain_skip_filter_by_picture_single_variant_attr",
            "param_type": "int",
            "report_ab_hit": "{{_ABTEST_REPORT_HIT_}}"
          },
          {
            "attr_name": "fountain_skip_filter_by_picture_variant_attr",
            "default_value": 1,
            "param_name": "fountain_skip_filter_by_picture_variant_attr",
            "param_type": "int",
            "report_ab_hit": "{{_ABTEST_REPORT_HIT_}}"
          },
          {
            "attr_name": "fountain_skip_filter_by_picture_set_variant_attr",
            "default_value": 0,
            "param_name": "fountain_skip_filter_by_picture_set_variant_attr",
            "param_type": "int",
            "report_ab_hit": "{{_ABTEST_REPORT_HIT_}}"
          },
          {
            "attr_name": "fountain_enable_skip_high_value_pic",
            "param_name": "fountain_enable_skip_high_value_pic",
            "default_value": 0,
            "param_type": "int",
          },
          {
            "attr_name": "fountain_short_duration_filter_limit",
            "param_name": "fountain_short_duration_filter_limit",
            "default_value": 3,
            "param_type": "int",
          },
          {
            "attr_name": "fountain_topk_audit_second_level_white_tags",
            "param_name": "fountain_topk_audit_second_level_white_tags",
            "default_value": "2008234,2008235,2008236",
            "param_type": "string",
          },
          {
            "attr_name": "fountain_topk_audit_second_level_black_tags",
            "param_name": "fountain_topk_audit_second_level_black_tags",
            "default_value": "",
            "param_type": "string",
          },
          {
            "attr_name": "fountain_topk_audit_bad_recall_filter",
            "param_name": "fountain_topk_audit_bad_recall_filter",
            "default_value": False,
            "param_type": "bool",
          },
          {
            "attr_name": "fountain_topk_audit_bad_recall_filter_use_global",
            "param_name": "fountain_topk_audit_bad_recall_filter_use_global",
            "default_value": False,
            "param_type": "bool",
          },
          {
            "attr_name": "fountain_topk_audit_bad_recall_filter_credible_ques_cnt",
            "param_name": "fountain_topk_audit_bad_recall_filter_credible_ques_cnt",
            "default_value": 30,
            "param_type": "int",
          },
          {
            "attr_name": "fountain_topk_audit_bad_recall_filter_pos_threshold",
            "param_name": "fountain_topk_audit_bad_recall_filter_pos_threshold",
            "default_value": 1.0,
            "param_type": "double",
          },
          {
            "attr_name": "fountain_topk_audit_bad_recall_filter_mode",
            "param_name": "fountain_topk_audit_bad_recall_filter_mode",
            "default_value": 0,
            "param_type": "int",
          },
          {
            "attr_name": "fountain_topk_audit_bad_recall_filter_unsure_threshold",
            "param_name": "fountain_topk_audit_bad_recall_filter_unsure_threshold",
            "default_value": 1.0,
            "param_type": "double",
          },
          {
            "attr_name": "fountain_topk_audit_bad_recall_filter_neg_threshold",
            "param_name": "fountain_topk_audit_bad_recall_filter_neg_threshold",
            "default_value": 1.0,
            "param_type": "double",
          },
          {
            "attr_name": "fountain_topk_audit_bad_recall_filter_hate_threshold",
            "param_name": "fountain_topk_audit_bad_recall_filter_hate_threshold",
            "default_value": 1.0,
            "param_type": "double",
          },
          {
            "attr_name": "fountain_enable_risk_man_risk_photo_filter",
            "param_name": "fountain_enable_risk_man_risk_photo_filter",
            "default_value": True,
            "param_type": "bool",
          },
          {
            "attr_name": "fountain_enable_audit_hack_photo_filter",
            "param_name": "fountain_enable_audit_hack_photo_filter",
            "default_value": False,
            "param_type": "bool",
          },
          {
            "attr_name": "audit_hack_tags_str",
            "param_name": "audit_hack_tags_str",
            "default_value": "2037808,2037809,2037810,2037811,2037812,2037813",
            "param_type": "string",
          },
          {
            "attr_name": "audit_hack_photo_filter_min_show",
            "param_name": "audit_hack_photo_filter_min_show",
            "default_value": 10000,
            "param_type": "int",
          },
          {
            "attr_name": "audit_hack_photo_filter_max_ltr",
            "param_name": "audit_hack_photo_filter_max_ltr",
            "default_value": 0.08,
            "param_type": "double",
          },
          {
            "attr_name": "audit_hack_photo_filter_max_wtr",
            "param_name": "audit_hack_photo_filter_max_wtr",
            "default_value": 0.01,
            "param_type": "double",
          },
          {
            "attr_name": "audit_hack_photo_filter_max_cmtr",
            "param_name": "audit_hack_photo_filter_max_cmtr",
            "default_value": 0.02,
            "param_type": "double",
          },
          {
            "attr_name": "fountain_data_set_tags_filter_tags_list_str",
            "param_name": "fountain_data_set_tags_filter_tags_list_str",
            "default_value": "",
            "param_type": "string",
          },
          {
            "attr_name": "fountain_quality_audit_filter_tags_list_str_final",
            "param_name": "fountain_quality_audit_filter_tags_list_str_final",
            "default_value": "2147250,2147252,2147253,2147254",
            "param_type": "string",
          },
          {
            "attr_name": "enable_fountain_video_filter",
            "default_value": False,
            "param_name": "enable_fountain_video_filter",
            "param_type": "bool",
            "report_ab_hit": "{{_ABTEST_REPORT_HIT_}}"
          },
          {
            "attr_name": "fountain_enable_explore_punish_city_filter",
            "param_name": "fountain_enable_explore_punish_city_filter",
            "default_value": 0,
            "param_type": "int",
          },
          {
            "attr_name": "enable_xlife_index_filter",
            "default_value": False,
            "param_name": "enable_xlife_index_filter",
            "param_type": "bool"
          },
          {
            "attr_name": "xlife_index_low_quality_filter_thresh_list_attr",
            "default_value": "0.85,0.88,0.94,0.91,0.96",
            "param_name": "xlife_index_low_quality_filter_thresh_list_attr",
            "param_type": "string"
          },
          {
            "attr_name": "xlife_index_low_quality_tag_list_attr",
            "default_value": "4009000,4009001,4009002,4009003,4009004",
            "param_name": "xlife_index_low_quality_tag_list_attr",
            "param_type": "string"
          },
          {
            "attr_name": "enable_life_fountain_author_filter",
            "default_value": False,
            "param_name": "enable_life_fountain_author_filter",
            "param_type": "bool"
          },
          {
            "attr_name": "life_fountain_author_grade_thresh",
            "default_value": 0,
            "param_name": "life_fountain_author_grade_thresh",
            "param_type": "int"
          },
          {
            "attr_name": "life_fountain_author_punish_cnt_mode",
            "default_value": 0,
            "param_name": "life_fountain_author_punish_cnt_mode",
            "param_type": "int"
          },
          {
            "attr_name": "life_fountain_author_filter_markcode",
            "default_value": "",
            "param_name": "life_fountain_author_filter_markcode",
            "param_type": "string"
          },
          {
            "attr_name": "life_fountain_author_punish_markcode",
            "default_value": "",
            "param_name": "life_fountain_author_punish_markcode",
            "param_type": "string"
          },
          {
            "attr_name": "enable_life_fountain_auto_audit_hot_cover_level_filter",
            "default_value": False,
            "param_name": "enable_life_fountain_auto_audit_hot_cover_level_filter",
            "param_type": "bool"
          },
          {
            "attr_name": "enable_life_fountain_auto_audit_follow_author_exemption",
            "default_value": 1,
            "param_name": "enable_life_fountain_auto_audit_follow_author_exemption",
            "param_type": "int"
          },
          {
            "attr_name": "enable_life_fountain_auto_audit_impression_good_ignore",
            "default_value": 0,
            "param_name": "enable_life_fountain_auto_audit_impression_good_ignore",
            "param_type": "int"
          },
          {
            "attr_name": "enable_life_fountain_auto_audit_bad_show_limit",
            "default_value": 0,
            "param_name": "enable_life_fountain_auto_audit_bad_show_limit",
            "param_type": "int"
          },
          {
            "attr_name": "life_fountain_enable_author_shop_score_filter",
            "default_value": False,
            "param_name": "life_fountain_enable_author_shop_score_filter",
            "param_type": "bool"
          },
          {
            "attr_name": "life_fountain_author_shop_score_filter_limit_count",
            "default_value": 4.4,
            "param_name": "life_fountain_author_shop_score_filter_limit_count",
            "param_type": "double"
          },
          {
            "attr_name": "life_fountain_enable_author_shop_zero_protect",
            "default_value": 1,
            "param_name": "life_fountain_enable_author_shop_zero_protect",
            "param_type": "int"
          },
          {
            "attr_name": "life_fountain_enable_author_goods_score_filter",
            "default_value": False,
            "param_name": "life_fountain_enable_author_goods_score_filter",
            "param_type": "bool"
          },
          {
            "attr_name": "life_fountain_author_goods_score_filter_limit_count",
            "default_value": 4.4,
            "param_name": "life_fountain_author_goods_score_filter_limit_count",
            "param_type": "double"
          },
          {
            "attr_name": "life_fountain_enable_author_shop_zero_protect",
            "default_value": 1,
            "param_name": "life_fountain_enable_author_shop_zero_protect",
            "param_type": "int"
          },
          ("life_fountain_enable_pic_sexy_filter", False),
          ("life_fountain_sexy_pic_max_cnt", 100000000),
          ("life_fountain_sexy_pic_cnt_mode", 0),
          ("life_fountain_enable_pic_bad_cover_filter", False),
          ("life_fountain_pic_bad_cover_tags_str", "2023746,2231037"),
          ("life_fountain_enable_pic_low_quality_filter", False),
          ("life_fountain_pic_low_quality_tag_str", "4009000,4009001,4009002,4009003,4009004,4009006,4009007"),
          ("life_fountain_pic_low_quality_filter_thresh_list_str", "-1,-1,-1,-1,-1,-1,-1"),
          ("life_fountain_enable_pic_low_cost_filter", False),
          ("life_fountain_low_cost_pic_max_cnt", 100000000),
          ("life_fountain_low_cost_pic_cnt_mode", 0),
          ("life_fountain_enable_pic_hack_act_filter", False),
          ("life_fountain_hack_act_pic_tags_str", "2037808,2037809,2037810,2037811,2037812,2037813"),
          ("life_fountain_hack_act_pic_types_str", "1"),
          ("life_fountain_hack_act_pic_max_cnt", 100000000),
          ("life_fountain_hack_act_pic_cnt_mode", 0),
          ("life_fountain_enable_pic_mmu_hetu_tag_filter", False),
          ("life_fountain_pic_filter_mmu_tag_prob_str", ""),
          ("life_fountain_pic_filter_mmu_tag_skip_hv_str", ""),
          ("life_fountain_pic_filter_mmu_tag_vv_thr_str", ""),
          ("life_fountain_enable_pic_data_set_tags_bit_filter", False),
          ("life_fountain_pic_filter_data_set_tags_bits_str", ""),
          ("life_fountain_pic_punish_data_set_tags_bits_str", ""),
          ("life_fountain_pic_skip_filter_mark_cod_str", ""),
          ("life_fountain_pic_punish_data_set_tags_bit_vv_thresh", 0),
          ("life_fountain_pic_punish_data_set_tags_bit_filter_prob", 0.0),
          ("life_fountain_enable_pic_secure_grade_filter", False),
          ("life_fountain_pic_secure_grade_filter_code_str", ""),
          ("life_fountain_enable_pic_author_filter", False),
          ("life_fountain_pic_author_grade_thresh", 0),
          ("life_fountain_pic_author_punish_cnt_mode", 0),
          ("life_fountain_pic_author_filter_markcode", ""),
          ("life_fountain_pic_author_punish_markcode", ""),
          ("life_fountain_enable_pic_ecology_high_report_filter", False),
          ("life_fountain_pic_ecology_high_report_rate_threshold", 0.000254),
          ("life_fountain_pic_ecology_high_report_count_threshold", 10),
          ("life_fountain_pic_ecology_high_report_fans_count_threshold", 1000000),
          ("life_fountain_enable_pic_ecology_high_neg_pos_rate_filter", False),
          ("life_fountain_pic_ecology_high_neg_pos_rate_threshold", 0.226),
          ("life_fountain_enable_pic_ecology_high_short_play_rate_filter", False),
          ("life_fountain_pic_ecology_high_short_play_rate_threshold", 0.5165),
          ("life_fountain_pic_ecology_neg_rate_threshold", 0.00136),
          ("life_fountain_enable_pic_ecology_mix_interact_rate_filter", False),
          ("life_fountain_pic_ecology_interact_rate_threshold", 0.0143),
          ("life_fountain_pic_ecology_interact_avg_view_time_threshold", 17.13),
          ("life_fountain_pic_ecology_interact_vv_threshold", 10000),
          ("life_fountain_enable_pic_mix_interact_rate_filter", False),
          ("life_fountain_pic_mix_interact_rate_filter_base_vv_threshold", 1000),
          ("life_fountain_pic_mix_interact_rate_filter_author_mark_cod_str", ""),
          ("life_fountain_pic_mix_interact_rate_thresholds_str", ""),
          ("life_fountain_pic_mix_interact_rate_filter_vv_thresholds_str", ""),
          ("life_fountain_pic_mix_interact_rate_filter_probs_str", ""),
          ("life_fountain_enable_marketing_static_video_filter", False),
          ("life_fountain_static_video_hetu_tag_id", 4009921),
          ("life_fountain_static_video_hetu_tag_prob_thd", 0.99),
          ("life_fountain_marketing_static_video_filter_mark_cod_str", ""),
          ("life_fountain_marketing_static_video_filter_base_vv_threshold", 1000),
          ("life_fountain_marketing_static_video_filter_interact_rate_thresholds_str", "0.0091,0.0292,1.0"),
          ("life_fountain_marketing_static_video_filter_vv_thresholds_str", "1000,5000,10000"),
          ("life_fountain_marketing_static_video_filter_probs_str", "1.0,0.7,0.5"),
        ]),
        prioritized_suffix="{{_ABTEST_SUFFIX_LIST_}}"
      ) \
      .split_string(
        input_common_attr = "xlife_index_low_quality_filter_thresh_list_attr",
        output_common_attr = "xlife_index_low_quality_filter_thresh",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_double = True
      ) \
      .split_string(
        input_common_attr = "xlife_index_low_quality_tag_list_attr",
        output_common_attr = "xlife_index_low_quality_tag",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_int = True
      ) \
      .split_string(
        input_common_attr="fountain_impression_audit_second_level_white_tags",
        output_common_attr="impression_audit_white_tags",
        delimiters=",",
        parse_to_int=True,
      )\
      .split_string(
        input_common_attr="fountain_impression_audit_second_level_black_tags",
        output_common_attr="impression_audit_black_tags",
        delimiters=",",
        parse_to_int=True,
      )\
      .split_string(
        input_common_attr="fountain_high_hot_audit_second_level_white_tags",
        output_common_attr="high_hot_audit_white_tags",
        delimiters=",",
        parse_to_int=True,
      )\
      .split_string(
        input_common_attr="fountain_high_hot_audit_second_level_black_tags",
        output_common_attr="high_hot_audit_black_tags",
        delimiters=",",
        parse_to_int=True,
      ) \
      .split_string(
        input_common_attr="fountain_topk_audit_second_level_white_tags",
        output_common_attr="topk_audit_white_tags",
        delimiters=",",
        parse_to_int=True,
      ) \
      .split_string(
        input_common_attr="fountain_topk_audit_second_level_black_tags",
        output_common_attr="topk_audit_black_tags",
        delimiters=",",
        parse_to_int=True,
      ) \
      .split_string(
        input_common_attr="fountain_data_set_tags_filter_tags_list_str",
        output_common_attr="data_set_tags_filter_tags_list",
        delimiters=",",
        trim_spaces=True,
        skip_empty_tokens=True,
        parse_to_int=True
      ) \
      .split_string(
        input_common_attr="fountain_quality_audit_filter_tags_list_str_final",
        output_common_attr="quality_audit_filter_tags_list",
        delimiters=",",
        trim_spaces=True,
        skip_empty_tokens=True,
        parse_to_int=True
      )\
      .if_("life_fountain_enable_pic_low_quality_filter == 1") \
        .split_string(
          input_common_attr = "life_fountain_pic_low_quality_tag_str",
          output_common_attr = "life_fountain_pic_low_quality_tag_list",
          delimiters = ",",
          trim_spaces = True,
          skip_empty_tokens = True,
          parse_to_int = True
        ) \
        .split_string(
          input_common_attr = "life_fountain_pic_low_quality_filter_thresh_list_str",
          output_common_attr = "life_fountain_pic_low_quality_filter_thresh_list",
          delimiters = ",",
          trim_spaces = True,
          skip_empty_tokens = True,
          parse_to_double = True
        ) \
      .end_() \
      .set_attr_value(
        common_attrs=[
          {
            "name": "single_picture_upload_type_list",
            "type": "int_list",
            "value": [7, 70],
          },
          {
            "name": "single_picture_picture_type_list",
            "type": "int_list",
            "value": [1],
          },
        ],
        skip="{{fountain_skip_filter_by_picture_single_variant_attr}}",
      ) \
      .set_attr_value(
        common_attrs=[
          {
            "name": "variant_picture_upload_type_list",
            "type": "int_list",
            "value": [10, 12],
          },
          {
            "name": "variant_picture_picture_type_list",
            "type": "int_list",
            "value": [3],
          },
        ],
        skip="{{fountain_skip_filter_by_picture_variant_attr}}",
      ) \
      .set_attr_value(
        common_attrs=[
          {
            "name": "variant_picture_set_upload_type_list",
            "type": "int_list",
            "value": [11],
          },
          {
            "name": "variant_picture_set_picture_type_list",
            "type": "int_list",
            "value": [2],
          },
        ],
        skip="{{fountain_skip_filter_by_picture_set_variant_attr}}",
      ) \
      .set_attr_value(
        common_attrs=[
          {
            "name": "long_article_upload_type_list",
            "type": "int_list",
            "value": [27],
          },
        ],
      ) \
      .set_attr_value(
        common_attrs=[
          {
            "name": "photo_life_max_hours",
            "type": "int",
            "value": 168,
          },
        ],
      )\
      .pack_common_attr(
        input_common_attrs=[
          "single_picture_picture_type_list",
          "variant_picture_picture_type_list",
          "variant_picture_set_picture_type_list"
        ],
        output_common_attr="filter_picture_type_list",
        deduplicate=False,
      ) \
      .pack_common_attr(
        input_common_attrs=[
          "single_picture_upload_type_list",
          "variant_picture_upload_type_list",
          "long_article_upload_type_list",
          "variant_picture_set_upload_type_list"
        ],
        output_common_attr="filter_upload_type_list",
        deduplicate=False,
      ) \
      .namespace_()
    return self

  @property
  def base_item_attr_map(self) -> dict:
    return {
      "photo_id_attr": "photo_id",
      "author_id_attr": "author__id",
      "photo_status_attr": "photo_status",
      "upload_time_attr": "upload_time",
      "upload_type_attr": "upload_type",
      "topk_audit_level_attr": "topk_audit_level",
      "topk_audit_tag_attr": "topk_audit_tag",
      "is_mid_video_photo_attr": "is_mid_video_photo",
      "questionaire_info_exposure_count_attr" : "questionnaire_info__exposure_count",
      "questionaire_info_negative_count_attr" : "questionnaire_info__negative_count",
      "questionaire_info_positive_count_attr" : "questionnaire_info__positive_count",
      "questionaire_info_unsure_count_attr" : "questionnaire_info__unsure_count",
      "explore_questionaire_info_exposure_count_attr" : "explore_questionnaire_info__exposure_count",
      "explore_questionaire_info_negative_count_attr" : "explore_questionnaire_info__negative_count",
      "explore_questionaire_info_positive_count_attr" : "explore_questionnaire_info__positive_count",
      "explore_questionaire_info_unsure_count_attr" : "explore_questionnaire_info__unsure_count",
      "level_hot_online_attr": "content_safety_level_with_namespace__level_hot_online",
      "audit_b_second_tag_attr": "audit_b_second_tag",
      "audit_hot_high_tag_level_attr": "audit_hot_high_tag_level",
      "risk_man_risk_photo_attr": "risk_level",
      "duration_ms_attr": "duration_ms",
      "long_term_photo_attr": "long_term_photo",
      "explore_punish_city_attr": "explore_punish_city",
      "explore_real_show_attr": "explore_stat__real_show_count",
      "explore_negative_attr": "explore_stat__negative_count",
      "explore_like_attr": "explore_stat__like_count",
      "explore_click_attr":"explore_stat__click_count",
      "photo_total_report_count_attr": "explore_stat__report_detail__total_report_count",
      "explore_follow_attr": "explore_stat__follow_count",
      "explore_forward_attr": "explore_stat__forward_count",
      "explore_comment_attr": "explore_stat__comment_count",
      "explore_collect_attr": "explore_stat__collect_count",
      "explore_view_length_sum_attr": "explore_stat__view_length_sum",
      "fountain_real_show_attr": "fountain_stats__real_show_count",
      "fountain_negative_attr": "fountain_stats__negative_count",
      "fountain_like_attr": "fountain_stats__like_count",
      "fountain_comment_attr": "fountain_stats__comment_count",
      "fountain_forward_attr": "fountain_stats__forward_count",
      "fountain_follow_attr": "fountain_stats__follow_count",
      "fountain_collect_attr": "fountain_stats__collect_count",
      "thanos_real_show_attr": "thanos_stats__real_show_count",
      "thanos_negative_attr": "thanos_stats__negative_count",
      "thanos_like_attr": "thanos_stats__like_count",
      "thanos_comment_attr": "thanos_stats__comment_count",
      "thanos_forward_attr": "thanos_stats__forward_count",
      "thanos_follow_attr": "thanos_stats__follow_count",
      "thanos_collect_attr": "thanos_stats__collect_count",
      "nebula_real_show_attr": "nebula_stats__real_show_count",
      "nebula_negative_attr": "nebula_stats__negative_count",
      "nebula_like_attr": "nebula_stats__like_count",
      "nebula_comment_attr": "nebula_stats__comment_count",
      "nebula_forward_attr": "nebula_stats__forward_count",
      "nebula_follow_attr": "nebula_stats__follow_count",
      "nebula_collect_attr": "nebula_stats__collect_count",
      "questionaire_info_negative_count_attr" : "questionnaire_info__negative_count",
      "questionaire_info_positive_count_attr" : "questionnaire_info__positive_count",
      "questionaire_info_unsure_count_attr" : "questionnaire_info__unsure_count",
      "topk_audit_level_attr": "topk_audit_level",
      "topk_audit_tag_attr": "topk_audit_tag",
      "picture_type_attr": "picture_type",
      "hetu_v2_level_one_tag_list_attr": "hetu_tag_level_info_v2__hetu_level_one",
      "hetu_v3_level_one_tag_list_attr": "hetu_tag_level_info_v3__hetu_level_one",
      "hetu_level_one_tag_list_attr": "hetu_tag_level_info__hetu_level_one",
      "width_attr": "width",
      "height_attr" : "height",
      "high_hot_audit_tag_v2_attr": "high_hot_audit_tag_v2",
      "merchant_item_id_list_attr": "merchant_item_info__item_id_list",
      "audit_user_experiment_level_attr": "audit_user_experiment_level",
      "eyeshot_source_attr": "eyeshot_source",
      "hetu_level_two_tag_list_attr": "hetu_tag_level_info__hetu_level_two",
      "hetu_level_three_tag_list_attr": "hetu_tag_level_info__hetu_level_three",
      "hetu_level_five_tag_list_attr": "hetu_tag_level_info__hetu_level_five",
      "hetu_tag_list_attr": "hetu_tag_level_info__hetu_tag",
      "hetu_face_id_tag_list_attr": "hetu_tag_level_info__hetu_face_id",
      "young_inc_tags_attr": "young_inc_tags",
      "final_cross_section_first_class_id_attr": "final_cross_section_first_class_id",
      "light_inc_photo_flag_attr": "light_inc_photo_flag",
      "author_fans_count_attr": "author__fans_count",
      "high_value_pic_flag_attr": "high_value_pic_flag",
      "data_set_tags_attr": "data_set_tags",
      "photo_dynamic_xtrs_str_attr": "video_cold_start_info__photo_dynamic_xtrs_str",
      "audit_risk_immd_tag_attr": "audit_risk_immd_tag",
      "data_set_tags_bit_attr": "data_set_tags_bit",
      "sirius_distribution_info__mark_cod_attr": "sirius_distribution_info__mark_cod",
      "author_grade_key_attr": "author_grade_key",
      "audit_hot_cover_level_attr": "audit_hot_cover_level",
      "hetu_tag_level_info_v2__hetu_tag_attr": "hetu_tag_level_info_v2__hetu_tag",
      "author_shop_score_attr":"author_shop_score",
      "author_max_item_score_attr":"author_max_item_score",
      "secure_grading_action_code_attr": "secure_grading_action_code",
      "explore_stats_report_count_attr": "explore_stat__report_count",
      "fountain_stats_report_count_attr": "fountain_stats__report_count",
      "thanos_stats_report_count_attr": "thanos_stats__report_count",
      "nebula_stats_report_count_attr": "nebula_stats__report_count",
      "explore_short_play_attr": "explore_stat__short_play_count",
      "fountain_stats_short_play_count_attr": "fountain_stats__short_play_count",
      "thanos_stats_short_play_count_attr": "thanos_stats__short_play_count",
      "nebula_stats_short_play_count_attr": "nebula_stats__short_play_count",
      "nebula_stats_view_length_sum_attr": "nebula_stats__view_length_sum",
      "thanos_stats_view_length_sum_attr": "thanos_stats__view_length_sum",
      "fountain_stats_view_length_sum_attr": "fountain_stats__view_length_sum",
    }

  @property
  def base_filters(self) -> list:
    return [
      {
        "name": "not_in_index",
        "enable": True,
      },
      {
        "name": "data_set_tags_filter",
        "enable": "{{fountain_enable_data_set_tags_filter}}",
        "filter_tags_list_attr": "data_set_tags_filter_tags_list"
      },
      {
        "name": "server_show_aid",
        "enable": True,
        "server_show_aid_list_attr": "browsedAuthorIds",
      },
      {
        "name": "photo_status",  # ？？
        "enable": True,
      },
      {
        "name": "over_180_days",
        "enable": True,
        "over_days_filter_days_limit_attr": "fountain_over_days_filter_days_limit",
        "entertainment_hetu_tags_attr": "fountain_entertainment_hetu_tag_str",
        "entertainment_hetu_days_limit_attr": "fountain_entertainment_hetu_days_limit_attr",
        "enable_filter_low_like": "enable_fountain_over_days_filter_low_like",
        "low_like_limit_attr": "fountain_over_days_filter_low_like_limit",
        "low_like_days_limit_attr": "fountain_over_days_filter_low_like_days_limit",
        "page_type": "FOUNTAIN",
        "topn_screen_filter_attr": "fountain_over_days_filter_topn_screen_map",
        "enable_filter_by_audit": "enable_fountain_over_days_filter_audit",
        "impression_audit_gray_hours_limit_attr": "fountain_impression_audit_gray_hours_limit",
        "impression_audit_normal_days_limit_attr": "fountain_impression_audit_normal_days_limit",
        "impression_audit_high_quality_days_limit_attr": "fountain_impression_audit_high_quality_days_limit",
        "high_hot_audit_gray_hours_limit_attr": "fountain_high_hot_audit_gray_hours_limit",
        "high_hot_audit_normal_days_limit_attr": "fountain_high_hot_audit_normal_days_limit",
        "high_hot_audit_high_quality_days_limit_attr": "fountain_high_hot_audit_high_quality_days_limit",
      },
      {
        "name": "impression_audit_bad",
        "enable": True,
        "impression_audit_white_tag_list_attr": "impression_audit_white_tags",
        "impression_audit_black_tag_list_attr": "impression_audit_black_tags",
        "level_hot_online_attr": "content_safety_level_with_namespace__level_hot_online",
        "audit_b_second_tag_attr": "audit_b_second_tag",
      },
      {
        "name": "high_hot_audit_bad",
        "enable": True,
        "high_hot_audit_white_tag_list_attr": "high_hot_audit_white_tags",
        "high_hot_audit_black_tag_list_attr": "high_hot_audit_black_tags",
        "audit_hot_high_tag_level_attr": "audit_hot_high_tag_level",
        "explore_operation_c_review_level_attr": "explore_operation_c_review_level",
      },
      {
        "name": "upload_type",
        "enable": "{{fountain_enable_upload_type_filter}}",
        "filter_type_list_attr": "filter_upload_type_list",
        "enable_skip_high_value_pic": "fountain_enable_skip_high_value_pic",
      },
      {
        "name": "picture_type",
        "enable": "{{fountain_enable_picture_type_filter}}",
        "filter_type_list_attr": "filter_picture_type_list",
        "enable_skip_high_value_pic": "fountain_enable_skip_high_value_pic",
      },
      {
        "name": "short_duration",
        "short_duration_limit_attr": "fountain_short_duration_filter_limit"
      },
      {
        "name": "topk_audit_bad",
        "topk_audit_white_tag_list_attr": "topk_audit_white_tags",
        "topk_audit_black_tag_list_attr": "topk_audit_black_tags",
        "topk_audit_bad_recall_filter_attr": "fountain_topk_audit_bad_recall_filter",
        "topk_audit_bad_recall_filter_use_global_attr": "fountain_topk_audit_bad_recall_filter_use_global",
        "topk_audit_bad_recall_filter_credible_ques_cnt_attr": "fountain_topk_audit_bad_recall_filter_credible_ques_cnt",
        "topk_audit_bad_recall_filter_pos_threshold_attr": "fountain_topk_audit_bad_recall_filter_pos_threshold",
        "topk_audit_bad_recall_filter_mode_attr": "fountain_topk_audit_bad_recall_filter_mode",
        "topk_audit_bad_recall_filter_unsure_threshold_attr": "fountain_topk_audit_bad_recall_filter_unsure_threshold",
        "topk_audit_bad_recall_filter_neg_threshold_attr": "fountain_topk_audit_bad_recall_filter_neg_threshold",
        "topk_audit_bad_recall_filter_hate_threshold_attr": "fountain_topk_audit_bad_recall_filter_hate_threshold",
      },
      {
        "name": "risk_man_risk_photo",
        "enable": "{{fountain_enable_risk_man_risk_photo_filter}}",
        "explore_user_risk_min_attr": "user_risk_min",
      },
      {
        "name": "photo_life",
        "enable": True,
        "photo_life_max_hours_attr": "photo_life_max_hours",
      },
      {  # 20 大
        "name": "explore_punish_city_filter",
        "enable": "{{fountain_enable_explore_punish_city_filter}}"
      },
      {
        "name": "black_author",
        "enable": True,
        "author_id_attr": "author__id",
      },
      {
        "name": "hate_author",
        "enable": True,
        "limit_hate_reason_attr": "fountain_limit_hate_reason",
      },
      # 诱导互动作品过滤
      {
        "name": "audit_hack_photo_filter",
        "enable": "{{fountain_enable_audit_hack_photo_filter}}",
        "audit_hack_tag_set_attr": "audit_hack_tags_str",
        "min_show_attr": "audit_hack_photo_filter_min_show",
        "max_ltr_attr": "audit_hack_photo_filter_max_ltr",
        "max_wtr_attr": "audit_hack_photo_filter_max_wtr",
        "max_cmtr_attr": "audit_hack_photo_filter_max_cmtr",
      },
      {
        "name": "quality_audit_filter",
        "enable": True,
        "filter_tags_list_attr": "quality_audit_filter_tags_list"
      },
      {
        "name": "xlife_index_filter",
        "enable": "{{enable_xlife_index_filter}}",
        "xlife_low_quality_filter_thresh_attr": "xlife_index_low_quality_filter_thresh",
        "xlife_low_quality_tag_list_attr": "xlife_index_low_quality_tag"
      },
      {
        "name": "video_filter",
        "enable": "{{enable_fountain_video_filter}}",
      },
      { # 负向作者过滤
        "name": "life_author_filter",
        "enable": "{{enable_life_fountain_author_filter}}",
        "author_grade_thresh_attr": "life_fountain_author_grade_thresh",
        "author_punish_cnt_mode_attr": "life_fountain_author_punish_cnt_mode",
        "author_filter_markcode_attr": "life_fountain_author_filter_markcode",
        "author_punish_markcode_attr": "life_fountain_author_punish_markcode"
      },
      { # 封面机审灰劣过滤
        "name": "auto_audit_hot_cover_level_filter",
        "enable": "{{enable_life_fountain_auto_audit_hot_cover_level_filter}}",
        "enable_follow_author_exemption_attr": "enable_life_fountain_auto_audit_follow_author_exemption",
        "enable_impression_good_ignore_attr": "enable_life_fountain_auto_audit_impression_good_ignore",
        "auto_audit_bad_show_limit_attr": "enable_life_fountain_auto_audit_bad_show_limit",
      },
       #  店铺分过滤
      {
        "name": "author_shop_score_filter",
        "enable": "{{life_fountain_enable_author_shop_score_filter}}",
        "author_shop_score_limit_attr": "life_fountain_author_shop_score_filter_limit_count",
        "author_shop_zero_protect_attr": "life_fountain_enable_author_shop_zero_protect"
      },
      #  商品分过滤
      {
        "name": "author_goods_score_filter",
        "enable": "{{life_fountain_enable_author_goods_score_filter}}",
        "author_goods_score_limit_attr": "life_fountain_author_goods_score_filter_limit_count",
        "author_goods_zero_protect_attr": "life_fountain_enable_author_goods_zero_protect"
      },
      # 性感图文过滤
      {
        "name": "pic_sexy_filter",
        "enable": "{{life_fountain_enable_pic_sexy_filter}}",
        "sexy_pic_max_cnt_attr": "life_fountain_sexy_pic_max_cnt",
        "sexy_pic_cnt_mode_attr": "life_fountain_sexy_pic_cnt_mode",
      },
      # 封面劣质图文过滤
      {
        "name": "pic_bad_cover_filter",
        "enable": "{{life_fountain_enable_pic_bad_cover_filter}}",
        "pic_bad_cover_tags_attr": "life_fountain_pic_bad_cover_tags_str"
      },
      # 低质量图文过滤
      {
        "name": "pic_low_quality_filter",
        "enable": "{{life_fountain_enable_pic_low_quality_filter}}",
        "pic_low_quality_filter_thresh_attr": "life_fountain_pic_low_quality_filter_thresh_list",
        "explore_pic_low_quality_tag_list_attr": "life_fountain_pic_low_quality_tag_list",
      },
      # 低成本图文过滤
      {
        "name": "pic_low_cost_filter",
        "enable": "{{life_fountain_enable_pic_low_cost_filter}}",
        "explore_low_cost_pic_max_cnt_attr": "life_fountain_low_cost_pic_max_cnt",
        "explore_low_cost_pic_cnt_mode_attr": "life_fountain_low_cost_pic_cnt_mode",
      },
      # 诱导互动图文过滤
      {
        "name": "pic_hack_act_filter",
        "enable": "{{life_fountain_enable_pic_hack_act_filter}}",
        "explore_hack_act_pic_tags_attr": "life_fountain_hack_act_pic_tags_str",
        "explore_hack_act_pic_types_attr": "life_fountain_hack_act_pic_types_str",
        "explore_hack_act_pic_max_cnt_attr": "life_fountain_hack_act_pic_max_cnt",
        "explore_hack_act_pic_cnt_mode_attr": "life_fountain_hack_act_pic_cnt_mode",
      },
      # 图文 mmu hetu tag 过滤
      {
        "name": "pic_mmu_hetu_tag_filter",
        "enable": "{{life_fountain_enable_pic_mmu_hetu_tag_filter}}",
        "mmu_tag_prob_str_attr": "life_fountain_pic_filter_mmu_tag_prob_str",
        "mmu_tag_skip_hv_str_attr": "life_fountain_pic_filter_mmu_tag_skip_hv_str",
        "mmu_tag_vv_thr_str_attr": "life_fountain_pic_filter_mmu_tag_vv_thr_str",
      },
      # 图文 data_set_tags_bit 过滤
      {
        "name": "pic_data_set_tags_bit_filter",
        "enable": "{{life_fountain_enable_pic_data_set_tags_bit_filter}}",
        "pic_filter_bits_str_attr": "life_fountain_pic_filter_data_set_tags_bits_str",
        "pic_punish_bits_str_attr": "life_fountain_pic_punish_data_set_tags_bits_str",
        "skip_filter_mark_cod_str_attr": "life_fountain_pic_skip_filter_mark_cod_str",
        "punish_vv_thresh_attr": "life_fountain_pic_punish_data_set_tags_bit_vv_thresh",
        "punish_filter_prob_attr": "life_fountain_pic_punish_data_set_tags_bit_filter_prob"
      },
      # 图文安全审过滤
      {
        "name": "pic_secure_grade_filter",
        "enable": "{{life_fountain_enable_pic_secure_grade_filter}}",
        "secure_grade_filter_code_attr": "life_fountain_pic_secure_grade_filter_code_str"
      },
      # 图文负向作者过滤
      {
        "name": "pic_author_filter",
        "enable": "{{life_fountain_enable_pic_author_filter}}",
        "author_grade_thresh_attr": "life_fountain_pic_author_grade_thresh",
        "author_punish_cnt_mode_attr": "life_fountain_pic_author_punish_cnt_mode",
        "author_filter_markcode_attr": "life_fountain_pic_author_filter_markcode",
        "author_punish_markcode_attr": "life_fountain_pic_author_punish_markcode"
      },
      # 图文生态负向特征过滤：高举报
      {
        "name": "pic_ecology_high_report_filter",
        "enable": "{{life_fountain_enable_pic_ecology_high_report_filter}}",
        "explore_pic_ecology_high_report_rate_threshold_attr": "life_fountain_pic_ecology_high_report_rate_threshold",
        "explore_pic_ecology_high_report_count_threshold_attr": "life_fountain_pic_ecology_high_report_count_threshold",
        "pic_ecology_high_report_fans_count_threshold_attr": "life_fountain_pic_ecology_high_report_fans_count_threshold"
      },
      # 图文生态负向特征过滤：高负正反馈率
      {
        "name": "pic_ecology_high_neg_pos_rate_filter",
        "enable": "{{life_fountain_enable_pic_ecology_high_neg_pos_rate_filter}}",
        "explore_pic_ecology_high_neg_pos_rate_threshold_attr": "life_fountain_pic_ecology_high_neg_pos_rate_threshold"
      },
      # 图文生态负向特征过滤：高短播
      {
        "name": "pic_ecology_high_short_play_rate_filter",
        "enable": "{{life_fountain_enable_pic_ecology_high_short_play_rate_filter}}",
        "explore_pic_ecology_high_short_play_rate_threshold_attr": "life_fountain_pic_ecology_high_short_play_rate_threshold",
        "explore_pic_ecology_neg_rate_threshold_attr": "life_fountain_pic_ecology_neg_rate_threshold"
      },
      # 图文生态负向特征过滤：综合互动相关
      {
        "name": "pic_ecology_mix_interact_rate_filter",
        "enable": "{{life_fountain_enable_pic_ecology_mix_interact_rate_filter}}",
        "pic_ecology_interact_rate_threshold_attr": "life_fountain_pic_ecology_interact_rate_threshold",
        "pic_ecology_interact_avg_view_time_threshold_attr": "life_fountain_pic_ecology_interact_avg_view_time_threshold",
        "pic_ecology_interact_vv_threshold_attr": "life_fountain_pic_ecology_interact_vv_threshold"
      },
      # 营销号单图低综合互动率过滤
      {
        "name": "pic_mix_interact_rate_filter",
        "enable": "{{life_fountain_enable_pic_mix_interact_rate_filter}}",
        "base_vv_threshold_attr": "life_fountain_pic_mix_interact_rate_filter_base_vv_threshold",
        "author_filter_mark_cod_str_attr": "life_fountain_pic_mix_interact_rate_filter_author_mark_cod_str",
        "interact_rate_thresholds_str_attr": "life_fountain_pic_mix_interact_rate_thresholds_str",
        "vv_thresholds_str_attr": "life_fountain_pic_mix_interact_rate_filter_vv_thresholds_str",
        "filter_probs_str_attr": "life_fountain_pic_mix_interact_rate_filter_probs_str",
      },
      # 营销号泛单图过滤
      {
        "name": "marketing_static_video_filter",
        "enable": "{{life_fountain_enable_marketing_static_video_filter}}",
        "static_video_tag_id_attr": "life_fountain_static_video_hetu_tag_id",
        "static_video_tag_prob_thd_attr": "life_fountain_static_video_hetu_tag_prob_thd",
        "marketing_mark_cod_str_attr": "life_fountain_marketing_static_video_filter_mark_cod_str",
        "base_vv_threshold_attr": "life_fountain_marketing_static_video_filter_base_vv_threshold",
        "interact_rate_thresholds_str_attr": "life_fountain_marketing_static_video_filter_interact_rate_thresholds_str",
        "vv_thresholds_str_attr": "life_fountain_marketing_static_video_filter_vv_thresholds_str",
        "filter_probs_str_attr": "life_fountain_marketing_static_video_filter_probs_str"
      },
    ]

  def filter_by_attr_with_perf(self, **kwargs):
    attr_name = "default_attr"
    if_skip = 0
    for key, value in kwargs.items():
      if key == "attr_name":
        attr_name= value
      elif key == "skip":
        if_skip = value

    self.filter_by_attr(
      **kwargs
    ) \
    .perflog_reason_count(
      check_point = "filter_by_" + attr_name,
      skip = if_skip
    )

    return self


  def filter_video(self):
    self.filter_by_attr(
      attr_name="is_picture",
      remove_if="==",
      compare_to=0,
      remove_if_attr_missing=True,
      cancel_num=20,
    )
  def _flow_end(self):
    self \
    .get_abtest_params(
      biz_name = "RECO_RPC",
      ab_params = [
        {
          "attr_name": "enable_xlife_fountain_not_life_filter",
          "default_value": False,
          "param_name": "enable_xlife_fountain_not_life_filter",
          "param_type": "bool"
        },
        {
          "attr_name": "xlife_fountain_not_life_filter_only_fast",
          "default_value": True,
          "param_name": "xlife_fountain_not_life_filter_only_fast",
          "param_type": "bool"
        },
        {
          "attr_name": "enable_life_target_hetu_new",
          "default_value": False,
          "param_name": "enable_life_target_hetu_new",
          "param_type": "bool"
        },
        {
          "attr_name": "life_target_hetu_version",
          "default_value": "v1",
          "param_name": "life_target_hetu_version",
          "param_type": "string"
        }
      ]
    ) \
    .if_("enable_xlife_fountain_not_life_filter == 1") \
      .if_("enable_life_target_hetu_new == 1") \
        .get_kconf_params(
          kconf_configs = [
            {
              "kconf_key": "reco.eyeshot.LifeTabTargetHetuL2Json",
              "json_path": "{{life_target_hetu_version}}",
              "export_common_attr": "target_hetu_l2_list"
            },
            {
              "kconf_key": "reco.eyeshot.LifeTabGrayHetuL2Json",
              "json_path": "{{life_target_hetu_version}}",
              "export_common_attr": "gray_hetu_l2_list"
            },
            {
              "kconf_key": "reco.eyeshot.LifeTabTargetHetuL1Json",
              "json_path": "{{life_target_hetu_version}}",
              "export_common_attr": "target_hetu_l1_list"
            },
            {
              "kconf_key": "reco.eyeshot.LifeTabGrayHetuL1Json",
              "json_path": "{{life_target_hetu_version}}",
              "export_common_attr": "gray_hetu_l1_list"
            },
          ]
        ) \
        .enrich_attr_by_light_function(
          import_item_attr = [
            "photo_id",
            "hetu_tag_level_info__hetu_level_one",
            "hetu_tag_level_info__hetu_level_two"
          ],
          import_common_attr = [
            "target_hetu_l2_list",
            "gray_hetu_l2_list",
            "target_hetu_l1_list",
            "gray_hetu_l1_list"
          ],
          export_item_attr = [
            "gray_target", # 灰度 + 非生活打散，生活设为pid，灰度 + 非生活设为1
            "not_life_target" # 非生活打散，灰度 + 生活设置为pid，非生活设为1
          ],
          function_name = "ContentControlDiversifyTagV2",
          class_name = "ExploreLifeLightFunctionSet"
        ) \
      .else_() \
        .get_kconf_params(
          kconf_configs = [{
            "kconf_key": "reco.eyeshot.LifeTabTargetHetu",
            "value_type": "list_int64",
            "export_common_attr": "target_hetu_list",
            "default_value": []
          }]
        ) \
        .get_kconf_params(
          kconf_configs = [{
            "kconf_key": "reco.eyeshot.LifeTabNotTargetHetu",
            "value_type": "list_int64",
            "export_common_attr": "not_target_hetu_list",
            "default_value": []
          }]
        ) \
        .enrich_attr_by_light_function(
          import_item_attr = [
            "photo_id",
            "hetu_tag_level_info__hetu_level_one",
            "hetu_tag_level_info__hetu_level_two"
          ],
          import_common_attr = [
            "target_hetu_list",
            "not_target_hetu_list"
          ],
          export_item_attr = [
            "gray_target", # 灰度 + 非生活打散，生活设为pid
            "not_life_target" # 非生活打散，灰度 + 生活设置为pid
          ],
          function_name = "ContentControlDiversifyTag",
          class_name = "ExploreLifeLightFunctionSet"
        ) \
      .end_() \
      .count_reco_result(
        save_count_to = "not_life_filter_count",
        target_item = {"not_life_target": 1}
      ) \
      .if_("xlife_fountain_not_life_filter_only_fast == 0 or request_type == \"fountain_fast_v1_life\"") \
        .filter_by_attr(
          attr_name = "not_life_target",
          remove_if = "==",
          compare_to = 1
        ) \
        .perflog_attr_value(
          check_point = "fountain",
          aggregator = "avg",
          common_attrs = [
            "not_life_filter_count"
          ],
        ) \
      .end_() \
    .end_()
    self._perf_result(
      attr_map={
        "is_picture": ["pic", "count"],
        "is_support_author_picture": ["sp_aid_pic", "count"],
        "high_value_pic_flag": ["high_value_pic", "count"]
      },
      perf_sampling_attr="_IS_PERF_SAMPLING_REQUEST_",
    )
    super()._flow_end()
