from cascading import CommonModule

class CascadingCalcPicModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .if_(
        # 采样请求已计算过 picture ，不再计算，v2 计算方式推全时应替换
        "not _IS_PERF_SAMPLING_REQUEST_ or _IS_PERF_SAMPLING_REQUEST_ == 0") \
        .if_("enable_calc_is_picture_v2 > 0") \
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
      .if_("enable_cascading_use_longpic_picset == 1") \
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
      .if_("enable_explore_pic_prerank_queue == 1 or enable_explore_pic_prerank_emp_calc == 1") \
        .enrich_attr_by_light_function(
          import_item_attr = [
            "explore_stat__real_show_count",
            "explore_stat__click_count",
            "explore_stat__like_count",
            "explore_stat__follow_count",
            "explore_stat__forward_count",
            "explore_stat__long_play_count",
            "explore_stat__short_play_count",
            "explore_stat__profile_enter_count",
            "explore_stat__negative_count",
            "explore_stat__comment_count",
            "explore_stat__view_length_sum",
            "is_picture",
          ],
          export_item_attr = [
            "empirical_ctr",
            "empirical_ltr",
            "empirical_wtr",
            "empirical_ftr",
            "empirical_lvtr",
            "empirical_svtr",
            "empirical_ptr",
            "empirical_htr",
            "empirical_cmtr",
            "empirical_watch_time",
          ],
          function_name = "McCalEmpiricalXtr",
          class_name = "ExploreLightFunctionSetV2",
          target_item = { "is_picture" : 1 },
        ) \
      .end_() \
      .if_("enable_explore_pic_prerank_rand_score == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
          ],
          export_item_attr = [
            "prerank_rand_score"
          ],
          function_name = "CalcPrerankRandScore",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {"is_picture": 1},
        ) \
      .end_() \
      .if_("enable_explore_pic_revisited_item == 1") \
        .enrich_attr_by_light_function(
          import_item_attr = [
            "revisited_rate_1d",
            "revisited_rate_3d",
            "revisited_rate_7d"
          ],
          export_item_attr = [
            "revisited_rate_1d",
            "revisited_rate_3d",
            "revisited_rate_7d"
          ],
          function_name = "RevisitedTransLessZero2Zero",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {"is_picture": 1},
        ) \
      .end_()