from cascading_v2 import CommonModule

class CascadingPrerankPhotoLabelModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self._is_picture_label()
    self._is_follow_author_label()
    self._is_low_cost_label()
    self._is_minority_label()
    self._is_key_target_hetu_pic_label()
    self._cluster_id_632_label()
    self._is_pic_search_cluster_label()
    self._is_pic_interest_cluster_label()
    self._is_pic_grouth_cluster_label()
    self._is_xhs_type_label()
    self._hetu_level_one_top1_tag()
    self._is_short_uninterested_photo_label()
    self._is_pic_double_valid_interest_cluster_label()
    self._is_same_author_tail_label()
    self._mounted_interest_cluster_id()
    self._gen_is_reason_top_photo()

  def post_process(self) -> None:
    self.flow \
      .if_("enable_explore_pic_cluster_counter > 0 or explore_need_traceback > 0") \
        .explore_pic_cluster_counter_enricher(
          save_pic_cluster_distr_str_attr = "retr_pic_cluster_distr_str",
          save_long_term_interest_cnt_attr = "retr_pic_long_term_interest_count",
          save_short_term_interest_cnt_attr = "retr_pic_short_term_interest_count",
          save_explore_interest_cnt_attr = "retr_pic_explore_interest_count",
          save_unknown_interest_cnt_attr = "retr_pic_unknown_interest_count",
          save_pic_cnt_attr = "retr_pic_count",
          save_hetu_cnt_attr = "retr_pic_hetu_count",
          long_term_interest_list_attr = "explore_pic_long_interest_list",
          short_term_interest_list_attr = "explore_pic_short_interest_list",
          explore_interest_list_attr = "explore_pic_explore_interest_list",
          hetu_list_attr = "hetu_tag_level_info__hetu_level_one",
          target_item = {"is_picture": 1}
        ) \
      .end_()

  def _is_picture_label(self) -> None:
    self.flow \
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
      )

  def _is_follow_author_label(self) -> None:
    self.flow \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "followAids", "as": "attr_list"}
        ],
        import_item_attr = [
          {"name": "author__id", "as": "attr"}
        ],
        export_item_attr = [
          {"name": "is_in_set", "as": "is_follow_author"}
        ],
        function_name = "AttrIsInSet",
        class_name = "ExploreLightFunctionSetV2",
      )

  def _is_low_cost_label(self) -> None:
    self.flow \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "negative_aid_set_ptr", "as": "aid_set_ptr"}
        ],
        import_item_attr = [
          "author__id"
        ],
        export_item_attr = [
          {"name": "is_target_photo", "as": "is_low_cost_photo"}
        ],
        function_name = "AidInSet",
        class_name = "ExploreLightFunctionSetV2",
      )

  def _is_minority_label(self) -> None:
    self.flow \
      .split_string(
        input_common_attr = "explore_minority_photo_tags_bits_list_str",
        output_common_attr = "explore_minority_photo_tags_bits_list",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_int = True,
      ) \
      .split_string(
        input_common_attr = "explore_minority_photo_manjiao_markcode_tags_str",
        output_common_attr = "explore_minority_photo_manjiao_markcode_tags",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_int = True,
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "explore_minority_photo_tags_bits_list", "as": "minority_photo_bits_list"},
          {"name": "explore_minority_photo_manjiao_markcode_tags", "as": "manjiao_markcode_tags"}
        ],
        import_item_attr = [
          "data_set_tags_bit",
          "manjiao_markcode"
        ],
        export_item_attr = [
          "is_minority_photo",
        ],
        function_name = "IsMinorityPhotoV2",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .if_("enable_explore_prev_items_gen_minority_photo == 1") \
        .enrich_attr_by_light_function(
          item_list_from_attr = "explore_recent_play_list",
          import_common_attr = [
            {"name": "explore_minority_photo_tags_bits_list", "as": "minority_photo_bits_list"},
            {"name": "explore_minority_photo_manjiao_markcode_tags", "as": "manjiao_markcode_tags"}
          ],
          import_item_attr = [
            "data_set_tags_bit",
            "manjiao_markcode"
          ],
          export_item_attr = [
            "is_minority_photo",
          ],
          function_name = "IsMinorityPhotoV2",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("explore_ranking_diversity_enable_standard_explore_realshow_pid_list == 1") \
        .enrich_attr_by_light_function(
          item_list_from_attr = "standard_explore_realshow_pid_list",
          import_common_attr = [
            {"name": "explore_minority_photo_tags_bits_list", "as": "minority_photo_bits_list"},
            {"name": "explore_minority_photo_manjiao_markcode_tags", "as": "manjiao_markcode_tags"}
          ],
          import_item_attr = [
            "data_set_tags_bit",
            "manjiao_markcode"
          ],
          export_item_attr = [
            "is_minority_photo",
          ],
          function_name = "IsMinorityPhotoV2",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_()

  def _is_key_target_hetu_pic_label(self) -> None:
    self.flow \
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
      )

  def _is_pic_search_cluster_label(self) -> None:
    self.flow \
      .if_("enable_explore_pic_search_candicate_expand_by_cluster == 1") \
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
      .end_()

  def _cluster_id_632_label(self) -> None:
    self.flow \
      .if_("enable_explore_calc_photo_cluster_id_632 == 1") \
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
      .end_()
      

  def _is_pic_interest_cluster_label(self) -> None:
    self.flow \
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
          ],
          function_name = "CalcPicInterestCluster",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {"is_picture": 1}
        ) \
      .end_()

  def _is_pic_grouth_cluster_label(self) -> None:
    self.flow \
      .if_("explore_enable_user_pic_growth_cluster_boost == 1 and ((uDoubleOutsideValidPicClusterCnt7dKV or 0) < explore_user_pic_growth_cluster_boost_interest_thresh)") \
        .enrich_attr_by_light_function(
          target_item = {"is_picture": 1},
          import_common_attr= [
            {"name": "uPicGrowthCidList", "as": "attr_list"},
          ],
          import_item_attr = [
            {"name": "cluster_id_632", "as": "attr"}
          ],
          export_item_attr = [
            {"name": "is_in_set", "as": "is_pic_growth_cluster"}
          ],
          function_name = "AttrIsInSet",
          class_name = "ExploreLightFunctionSetV2"
        ) \
      .end_()
    
  def _is_xhs_type_label(self) -> None:
    self.flow \
      .if_("enable_calc_xhs_type_picture_v2 == 1") \
        .enrich_attr_by_light_function(
          target_item = { "is_picture": 1 },
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
      .end_()

  def _hetu_level_one_top1_tag(self) -> None:
    self.flow \
      .enrich_attr_by_light_function(
        import_item_attr = [
          {"name": "hetu_tag_level_info__hetu_level_one", "as": "extract_hetu_tag_list"},
        ],
        export_item_attr = [
          {"name": "first_hetu_tag", "as": "hetu_level_one_top1"},
        ],
        function_name = "ExtractFirstHetuTag",
        class_name = "ExploreLightFunctionSetV2",
      )

  def _is_short_uninterested_photo_label(self) -> None:
    self.flow \
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
          output_short_uninterest_hetu5_stat_attr = "short_uninterest_hetu5_stat"
        ) \
      .end_()

  def _is_pic_double_valid_interest_cluster_label(self) -> None:
    self.flow \
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
      .end_()

  def _is_same_author_tail_label(self) -> None:
    self.flow \
      .if_("explore_enable_gen_same_author_tail == 1") \
        .enrich_attr_by_light_function(
          import_item_attr = [
            {"name": "video_cold_start_info__explore_author_exp_tail", "as": "item_attr"}
          ],
          import_common_attr = [
            {"name": "explore_author_exp_tail", "as": "common_attr"}
          ],
          export_item_attr = [
            {"name": "judge", "as": "is_same_author_tail"},
          ],
          function_name = "JudgeItemAttrAndCommonAttrEqual",
          class_name = "ExploreLightFunctionSetV2"
        ) \
      .end_()

  def _mounted_interest_cluster_id(self):
    self.flow \
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
      .end_()

  def _gen_is_reason_top_photo(self):
    self.flow \
      .if_("explore_enable_gen_is_reason_top_photo == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_reason_top_photo_white_list", "as": "reason_white_list"},
            {"name": "explore_reason_top_photo_top_k", "as": "top_k"},
          ],
          export_item_attr = [
            {"name": "is_reason_top_photo", "as": "is_directly_reach_fullrank"},
          ],
          function_name = "CalIsReasonTopPhoto",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_()
 
 