#!/usr/bin/env python3
# coding=utf-8

from cascading import CommonModule

class CascadingPhotoTypeModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .if_("enable_sim_hetu_cluster_id_lv1_trans == 1") \
        .get_kconf_params(
          kconf_configs = [{
            "kconf_key": "reco.offline.cidGroupMapStr2Int",
            "json_path": "{{hetu_sim_cluster_id}}",
            "default_value": -1,
            "export_item_attr": "hetu_sim_cluster_id862_lv1",
          }]
        ) \
      .end_() \
      .if_("enable_explore_calc_photo_cluster_id_632 == 1 or is_traceback_request == 1") \
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
        ) \
        .enrich_attr_by_light_function(
          item_list_from_attr = "explore_realshow_pids",
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
        .if_("explore_partial_time_based_interest_adjust == 1") \
          .enrich_attr_by_light_function(
            item_list_from_attr = "partial_time_based_selected_pids",
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
        .if_("enable_cal_standard_explore_realshow_list_cluster_id_632 == 1") \
          .enrich_attr_by_light_function(
            item_list_from_attr = "standard_explore_realshow_pid_list",
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
        .if_("enable_cal_explore_user_recent_click_list_cluster_id_632 == 1") \
          .enrich_attr_by_light_function(
            item_list_from_attr = "explore_user_recent_click_list",
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
        .if_("enable_cal_explore_user_recent_play_list_cluster_id_632 == 1") \
          .enrich_attr_by_light_function(
            item_list_from_attr = "explore_recent_play_list",
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
        .if_("explore_ranking_diversity_enable_standard_explore_realshow_pid_list == 1") \
          .enrich_attr_by_light_function(
            item_list_from_attr = "standard_explore_realshow_pid_list",
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
        .if_("explore_enable_interest_score_based_pids_cid_632 == 1 and interest_score_based_valid_user == 1") \
          .enrich_attr_by_light_function(
            item_list_from_attr = "interest_score_based_pids",
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
      .end_() \
      .if_("enable_explore_frist_screen_customization_use_interest_cluster_id_632 == 1 and is_first_refresh == 1") \
        .explore_frist_screen_customization_use_interest_cluster_id_632() \
      .end_() \
      .if_("enable_explore_use_interest_cluster_id_632 == 1") \
        .copy_attr(
          attrs=[{
            "from_item": "cluster_id_632",
            "to_item": "interest_cluster_id"
          }]
        ) \
      .else_() \
        .copy_attr(
          attrs=[{
            "from_item": "hetu_sim_cluster_id",
            "to_item": "interest_cluster_id"
          }]
        ) \
      .end_() \
      .if_("enable_explore_use_mounted_interest_cluster_id == 1") \
        .copy_attr( # mounted_interest_cluster_id 用于 fix 挂载兴趣 cid 版本和候选视频 cid 标签版本不对齐的问题, 实验后若能推全将会统一使用interest_cluster_id
          attrs=[{
            "from_item": "cluster_id_632",
            "to_item": "mounted_interest_cluster_id"
          }]
        ) \
      .else_() \
        .copy_attr(
          attrs=[{
            "from_item": "hetu_sim_cluster_id",
            "to_item": "mounted_interest_cluster_id"
          }]
        ) \
      .end_() \
      .if_("enable_explore_recent_play_list_use_mounted_interest_cluster_id == 1") \
        .copy_attr( # mounted_interest_cluster_id 用于 fix 挂载兴趣 cid 版本和候选视频 cid 标签版本不对齐的问题, 实验后若能推全将会统一使用interest_cluster_id
          item_list_from_attr = "explore_recent_play_list",
          attrs=[{
            "from_item": "cluster_id_632",
            "to_item": "mounted_interest_cluster_id"
          }]
        ) \
        .if_("explore_ranking_diversity_enable_standard_explore_realshow_pid_list == 1") \
          .copy_attr( # mounted_interest_cluster_id 用于 fix 挂载兴趣 cid 版本和候选视频 cid 标签版本不对齐的问题, 实验后若能推全将会统一使用interest_cluster_id
            item_list_from_attr = "standard_explore_realshow_pid_list",
            attrs=[{
              "from_item": "cluster_id_632",
              "to_item": "mounted_interest_cluster_id"
            }]
          ) \
        .end_() \
      .else_() \
        .copy_attr(
          item_list_from_attr = "explore_recent_play_list",
          attrs=[{
            "from_item": "hetu_sim_cluster_id",
            "to_item": "mounted_interest_cluster_id"
          }]
        ) \
        .if_("explore_ranking_diversity_enable_standard_explore_realshow_pid_list == 1") \
          .copy_attr(
            item_list_from_attr = "standard_explore_realshow_pid_list",
            attrs=[{
              "from_item": "hetu_sim_cluster_id",
              "to_item": "mounted_interest_cluster_id"
            }]
          ) \
        .end_() \
      .end_() \
      .if_("enable_explore_calc_new_interest_explore_type == 1") \
        .gen_is_new_interest_explore() \
      .end_() \
      .if_("explore_enable_all_page_valid_interest_tagger == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "uPicValidInterestClusterIdList", "as": "attr_list"},
            {"name": "cluster_id_632_default_value", "as": "default_value"},
          ],
          import_item_attr = [
            {"name": "cluster_id_632", "as": "attr"}
          ],
          export_item_attr = [
            {"name": "is_in_set", "as": "is_all_page_valid_interest"}
          ],
          function_name = "AttrIsInSet",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("enable_explore_calc_is_user_short_develop_interest == 1 and uExploreFountainPreferenceTypeKV ~= nil and uExploreFountainPreferenceTypeKV == 1") \
        .gen_user_develop_interest_score() \
      .end_() \
      .if_("enable_unbias_interest_selected_cids_photo == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "unbias_interest_selected_cids", "as": "attr_list"},
          ],
          import_item_attr = [
            {"name" : "hetu_sim_cluster_id", "as" : "attr"}
          ],
          export_item_attr = [
            {"name": "is_in_set", "as": "is_in_selected_cids"}
          ],
          function_name = "AttrIsInSet",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("enable_short_uninterest_tagger == 1") \
        .explore_short_uninterest_tagger(
          prev_item_from_attr = "standard_explore_realshow_pid_list",
          prev_item_from_attr_timestamp = "uStandardExploreRealshowTimestampList",
          prev_item_label_from_attr = "uStandardExploreRealshowLabelList",
          time_window = "{{explore_short_unterested_timestamp_threshold}}",
          realshow_num_threshold = "{{explore_short_unterested_realshow_num_threshold}}",
          realshow_no_click_threshold = "{{explore_short_unterested_no_click_threshold}}",
          cluster_id_attr = "cluster_id_632",
          hetu_level_five_attr = "hetu_tag_level_info__hetu_level_five",
          enable_cluster_id = "{{explore_short_unterested_enable_cluster_id}}",
          enable_hetu_level_five = "{{explore_short_unterested_enable_hetu_level_five}}",
          output_short_uninterest_flag_attr = "is_short_uninterested_photo",
          output_short_uninterest_cid_num_attr = "short_uninterest_cid_num",
          output_short_uninterest_hetu5_num_attr = "short_uninterest_hetu5_num",
          output_short_uninterest_cid_stat_attr = "short_uninterest_cid_stat",
          output_short_uninterest_hetu5_stat_attr = "short_uninterest_hetu5_stat",
        ) \
      .end_() \
      .if_("explore_enable_adjust_marketing_compensation_photo == 1") \
        .gen_is_marketing_compensation_photo() \
      .end_() \
      .if_("explore_enable_outer_field_interest_photo == 1") \
        .gen_is_outer_field_interest_photo() \
      .end_() \
      .if_("explore_enable_gen_hetu_first_tag == 1") \
        .gen_hetu_first_tag() \
      .end_() \
      .if_("explore_enable_adjust_protogenetic_advertise_photo == 1") \
        .gen_is_protogenetic_advertise_photo() \
      .end_() \
      .if_("explore_enable_gen_is_olympic_photo == 1") \
        .gen_is_olympic_photo() \
      .end_() \
      .if_("explore_enable_gen_photo_show_ration == 1") \
        .gen_photo_show_ration() \
      .end_() \
      .if_("explore_enable_gen_is_low_cost_photo == 1") \
        .gen_is_low_cost_photo() \
      .end_() \
      .if_("enable_explore_gen_minority_photo_v2 == 1") \
        .gen_is_minority_photo() \
      .end_() \
      .if_("explore_enable_gen_is_top_author_new_photo == 1")\
        .gen_is_top_audit_photo() \
      .end_() \
      .if_("explore_enable_gen_is_sexy_induce_photo == 1")\
        .gen_is_sexy_induce_photo() \
      .end_() \
      .if_("explore_enable_gen_author_circle_cluster_id == 1") \
        .gen_u2a_author_circle_cluster_id() \
      .end_() \
      .if_("enable_explore_gen_is_meinv_photo == 1") \
        .gen_is_meinv_photo() \
      .end_() \
      .if_("enable_explore_gen_upload_time_day == 1") \
        .gen_upload_time_day() \
      .end_() \
      .if_("enable_explore_gen_upload_time_second == 1") \
        .gen_upload_time_second() \
      .end_() \
      .if_("explore_enable_gen_is_new_hot_photo == 1")\
        .gen_is_new_hot_photo() \
      .end_() \
      .if_("enable_explore_transform_photo_proinc_type == 1") \
        .item_attr_operation(
          item_attr_a = "photo_proinc_type",
          common_attr_b = 8,
          operator = "&",
          output_attr = "userfulness_author_tag"
        ) \
        .cast_attr_type(
          attr_type_cast_configs = [
            {
              "to_type": "double",
              "from_item_attr": "userfulness_author_tag",
              "to_item_attr": "userfulness_author_score"
            }
          ]
        ) \
      .end_() \
      .if_("enable_explore_transform_photo_proinc_type_to_authority_tag == 1") \
        .item_attr_operation(
          item_attr_a = "photo_proinc_type",
          common_attr_b = 16,
          operator = "&",
          output_attr = "authority_author_tag"
        ) \
        .cast_attr_type(
          attr_type_cast_configs = [
            {
              "to_type": "double",
              "from_item_attr": "authority_author_tag",
              "to_item_attr": "authority_author_score"
            }
          ]
        ) \
      .end_() \
      .if_("enable_explore_transform_photo_proinc_type_to_expertise_tag == 1") \
        .item_attr_operation(
          item_attr_a = "photo_proinc_type",
          common_attr_b = 64,
          operator = "&",
          output_attr = "expertise_author_tag"
        ) \
        .cast_attr_type(
          attr_type_cast_configs = [
            {
              "to_type": "double",
              "from_item_attr": "expertise_author_tag",
              "to_item_attr": "expertise_author_score"
            }
          ]
        ) \
      .end_() \
      .split_string(
        input_common_attr = "exclude_hetu_level_one_tag_list_str",
        output_common_attr = "exclude_hetu_level_one_tag_list",
        delimiters = ",",
        parse_to_int = True
      ) \
      .if_("enable_explore_gen_photo_original_submission_tag == 1") \
        .enrich_attr_by_light_function(
          import_item_attr = [
            "sirius_distribution_info__mark_cod",
            "hetu_tag_level_info__hetu_level_one",
            "author__id"
          ],
          import_common_attr = [
            {"name": "enable_explore_original_submission_combine_exclusion", "as": "enable_exclusion"},
            "kuaishou_official_account_set_ptr",
            "exclude_hetu_level_one_tag_list"
          ],
          export_item_attr = [
            "original_submission_author_tag",
          ],
          function_name = "GenPhotoOriginalSubmissionTag",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("enable_explore_gen_photo_personalization_author_tag == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "personalization_author_set_ptr",
            {"name": "enable_explore_personalization_combine_exclusion", "as": "enable_exclusion"},
            "kuaishou_official_account_set_ptr",
            "exclude_hetu_level_one_tag_list"
          ],
          import_item_attr = [
            "author__id",
            "hetu_tag_level_info__hetu_level_one",
          ],
          export_item_attr = [
            "personalization_author_tag",
          ],
          function_name = "GenPhotoPersonalizationTag",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("explore_enable_gen_is_good_author_pool_photo == 1") \
        .gen_is_good_author_pool_photo() \
      .end_() \
      .if_("explore_enable_gen_is_ugc_photo == 1") \
        .gen_is_ugc_photo() \
      .end_() \
      .if_("explore_enable_gen_is_reason_top_photo == 1") \
        .gen_is_reason_top_photo() \
      .end_() \
      .if_("explore_enable_gen_is_first_refresh_good_photo == 1") \
        .gen_is_first_refresh_good_photo() \
      .end_() \
      .if_("explore_enable_gen_lowvv_tag == 1") \
        .gen_lowvv_tag() \
      .end_() \
      .if_("explore_enable_gen_same_author_tail == 1") \
        .gen_same_author_tail_tag() \
      .end_() \
      .if_("explore_enable_gen_photo_quality_score == 1") \
        .gen_photo_quality_score() \
      .end_() \
      .if_("explore_enable_gen_is_picture_follow_author == 1") \
        .gen_is_picture_follow_author() \
      .end_() \

