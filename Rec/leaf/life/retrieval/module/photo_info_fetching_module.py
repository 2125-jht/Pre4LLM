from retrieval import CommonModule

class PhotoInfoFetchingModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  # explore_request_data_set_tags 的取值请参考 enum DataSetTrialTag 的定义
  # https://kdev.corp.kuaishou.com/git/community-science/kuaishou-ad-reco-base-proto/-/file-detail?branchName=master&filePath=src/main/proto/kuaishou/newsmodel/reco_base.proto&repoId=17967&repoName=kuaishou-ad-reco-base-proto
  # 默认值为 "1,6" 表示获取 DATA_SET_TAG_BASE 和 DATA_SET_TAG_7DAYS 数据集合
  def process(self):
    self.flow \
      .get_abtest_params(
        biz_name="RECO_RPC",
        ab_params=[
          {
            "attr_name": "explore_request_data_set_tags",
            "default_value": "1,6",
            "param_name": "explore_request_data_set_tags",
            "param_type": "string",
            "report_ab_hit": "{{_ABTEST_REPORT_HIT_}}"
          },
          {
            "param_name": "fountain_request_data_set_tags_all",
            "param_type": "string",
            "attr_name": "fountain_request_data_set_tags_all",
            "default_value": ""
          },
          {
            "param_name": "xlife_fountain_request_data_set_tags",
            "param_type": "string",
            "attr_name": "xlife_fountain_request_data_set_tags",
            "default_value": "9,33",
          },
          {
            "attr_name": "life_use_hetu_v3",
            "default_value": True,
            "param_name": "life_use_hetu_v3",
            "param_type": "bool"
          },
          {
            "attr_name": "enable_life_fetch_realshow_photo_info",
            "default_value": False,
            "param_name": "enable_life_fetch_realshow_photo_info",
            "param_type": "bool"
          },
        ],
        prioritized_suffix="{{_ABTEST_SUFFIX_LIST_}}"
      ) \
      .copy_item_meta_info(
        save_item_id_to_attr="item_id"
      ) \
      .if_("life_use_hetu_v3 == 1") \
        .switch_("request_type") \
          .case_("fountain_splash_life") \
            .get_item_attr_by_distributed_flat_index(
              photo_store_kconf_key="reco.distributedIndex.hotPhotoInfoCommonIndex",
              perf_log="entry_photo_splash",
              photo_store_request_data_set_tags_attr="fountain_request_data_set_tags_all",
              use_dynamic_photo_store=True,
              # 获取入口 photo，不对索引命中率进行限制
              attrs=self.hetu_v3_source_photo_base_attrs_splash,
              additional_item_source={
                "reco_results": False,
                "common_attr": ["featureSourcePId"]
              }
            ) \
            .get_item_attr_by_distributed_flat_index(
              photo_store_kconf_key="reco.distributedIndex.hotPhotoInfoCommonIndex",
              perf_log="retr_index_splash",
              photo_store_rpc_req_cache_rate=100.0,
              photo_store_request_data_set_tags_attr="xlife_fountain_request_data_set_tags",
              use_dynamic_photo_store=True,
              item_id_attr="item_id",
              attrs=self.hetu_v3_attr_config,
            ) \
          .case_("fountain_splash_life_pic_inside") \
            .get_item_attr_by_distributed_flat_index(
              photo_store_kconf_key="reco.distributedIndex.hotPhotoInfoCommonIndex",
              perf_log="entry_photo_splash",
              photo_store_request_data_set_tags_attr="fountain_request_data_set_tags_all",
              use_dynamic_photo_store=True,
              # 获取入口 photo，不对索引命中率进行限制
              attrs=self.hetu_v3_source_photo_base_attrs_splash,
              additional_item_source={
                "reco_results": False,
                "common_attr": ["featureSourcePId"]
              }
            ) \
            .get_item_attr_by_distributed_flat_index(
              photo_store_kconf_key="reco.distributedIndex.hotPhotoInfoCommonIndex",
              perf_log="retr_index_splash",
              photo_store_rpc_req_cache_rate=100.0,
              photo_store_request_data_set_tags_attr="xlife_fountain_request_data_set_tags",
              use_dynamic_photo_store=True,
              item_id_attr="item_id",
              attrs=self.hetu_v3_attr_config,
            ) \
          .case_("fountain_fast_v1_life") \
            .get_item_attr_by_distributed_flat_index(
              photo_store_kconf_key="reco.distributedIndex.hotPhotoInfoCommonIndex",
              perf_log="retr_index",
              photo_store_rpc_req_cache_rate=100.0,
              photo_store_request_data_set_tags_attr="xlife_fountain_request_data_set_tags",
              use_dynamic_photo_store=True,
              item_id_attr="item_id",
              attrs=self.hetu_v3_photo_base_attrs_fast
            ) \
          .default_() \
            .get_item_attr_by_distributed_flat_index(
              photo_store_kconf_key="reco.distributedIndex.hotPhotoInfoCommonIndex",
              use_dynamic_photo_store=True,
              photo_store_request_data_set_tags_attr='explore_request_data_set_tags',
              attrs=self.hetu_v3_attr_config
            ) \
        .end_() \
      .else_() \
        .switch_("request_type") \
          .case_("fountain_splash_life") \
            .get_item_attr_by_distributed_flat_index(
              photo_store_kconf_key="reco.distributedIndex.hotPhotoInfoCommonIndex",
              perf_log="entry_photo_splash",
              photo_store_request_data_set_tags_attr="fountain_request_data_set_tags_all",
              use_dynamic_photo_store=True,
              # 获取入口 photo，不对索引命中率进行限制
              attrs=self.source_photo_base_attrs_splash,
              additional_item_source={
                "reco_results": False,
                "common_attr": ["featureSourcePId"]
              }
            ) \
            .get_item_attr_by_distributed_flat_index(
              photo_store_kconf_key="reco.distributedIndex.hotPhotoInfoCommonIndex",
              perf_log="retr_index_splash",
              photo_store_rpc_req_cache_rate=100.0,
              photo_store_request_data_set_tags_attr="xlife_fountain_request_data_set_tags",
              use_dynamic_photo_store=True,
              item_id_attr="item_id",
              attrs=self.base_attr_config,
            ) \
          .case_("fountain_splash_life_pic_inside") \
            .get_item_attr_by_distributed_flat_index(
              photo_store_kconf_key="reco.distributedIndex.hotPhotoInfoCommonIndex",
              perf_log="entry_photo_splash",
              photo_store_request_data_set_tags_attr="fountain_request_data_set_tags_all",
              use_dynamic_photo_store=True,
              # 获取入口 photo，不对索引命中率进行限制
              attrs=self.source_photo_base_attrs_splash,
              additional_item_source={
                "reco_results": False,
                "common_attr": ["featureSourcePId"]
              }
            ) \
            .get_item_attr_by_distributed_flat_index(
              photo_store_kconf_key="reco.distributedIndex.hotPhotoInfoCommonIndex",
              perf_log="retr_index_splash",
              photo_store_rpc_req_cache_rate=100.0,
              photo_store_request_data_set_tags_attr="xlife_fountain_request_data_set_tags",
              use_dynamic_photo_store=True,
              item_id_attr="item_id",
              attrs=self.base_attr_config,
            ) \
          .case_("fountain_fast_v1_life") \
            .get_item_attr_by_distributed_flat_index(
              photo_store_kconf_key="reco.distributedIndex.hotPhotoInfoCommonIndex",
              perf_log="retr_index",
              photo_store_rpc_req_cache_rate=100.0,
              photo_store_request_data_set_tags_attr="xlife_fountain_request_data_set_tags",
              use_dynamic_photo_store=True,
              item_id_attr="item_id",
              attrs=self.photo_base_attrs_fast
            ) \
          .default_() \
            .if_("enable_life_fetch_realshow_photo_info == 1") \
              .get_item_attr_by_distributed_flat_index(
                photo_store_kconf_key = "reco.distributedIndex.hotPhotoInfoCommonIndex",
                use_dynamic_photo_store = True,
                photo_store_request_data_set_tags_attr = 'explore_request_data_set_tags',
                attrs = self.base_attr_config,
                additional_item_source={
                  "reco_results": True,
                  "common_attr": ["life_realshow_common_list"]
                }
              ) \
            .else_() \
              .get_item_attr_by_distributed_flat_index(
                photo_store_kconf_key="reco.distributedIndex.hotPhotoInfoCommonIndex",
                use_dynamic_photo_store=True,
                photo_store_request_data_set_tags_attr='explore_request_data_set_tags',
                attrs=self.base_attr_config
              ) \
            .end_() \
        .end_() \
      .end_()
    self.enrich_picture_attrs()  # 填充图文相关的attr

    # 下面代码是临时hack下没有或没来源的字段，后续会删除
    self.flow.log_debug_info(
        item_attrs = [
          "photo_id",
          "author__id",
          "duration_ms",
          "upload_type",
          "upload_time",
          "author_age_info__age_segment",
          "is_merchant",
          "is_tnu_extend_index_photo",
          "cuckoo_info__is_cuckoo_photo",
          "percent_punish__data",
          "percent_punish__enabel_percent",
          "user_hash_tag_id",
          "photo_picture_count",
          "mmu_content_ids_33",
          "picture_type",
          "hetu_tag_level_info__hetu_cluster_id",
          "hetu_tag_level_info_v2__hetu_level_two",
          "is_hot_rank_photo",
          "data_set_tags",
          "collect_count",
          "cold_start_breakout_score",
          "audit_b_second_tag",
          "questionnaire_info__exposure_count",
          "questionnaire_info__negative_count",
          "questionnaire_info__positive_count",
          "questionnaire_info__unsure_count",
          "explore_questionnaire_info__exposure_count",
          "explore_questionnaire_info__negative_count",
          "explore_questionnaire_info__positive_count",
          "explore_questionnaire_info__unsure_count",
          "explore_punish_city",
          "explore_punish",
          "author__upload_count",
          "follow_count",
          "light_inc_photo_flag",
          "report_detail__low_report_count",
          "report_detail__total_report_count",
          "high_value_pic_flag",
          'fountain_stats__negative_count',
          'thanos_stats__negative_count',
          'nebula_stats__negative_count',
          'hetu_tag_level_info_v2__hetu_level_four',
          'hetu_tag_level_info_v2__hetu_face_id',
          'hetu_tag_level_info_v2__hetu_cluster_id',
          'online_lda_topic__ids',
          'author__category_detail__first_level_id',
          'author__category_detail__second_level_id',
          'author__category_detail__third_level_id',
          'author__category_detail__fourth_level_id',
          'GE_cluster_id',
          'mmu_text_cluster',
          'mmu_text_lda_topic',
          'is_mid_video_photo',
          'nearby_feeling__biz_name',
          'nearby_feeling__poi_info__poi_city_name',
          'nearby_feeling__poi_info__poi_district_name',
          'author__category_detail__third_level_id@featureSourcePId',
          'data_set_tags@featureSourcePId',
          'hetu_tag_level_info__hetu_cluster_id@featureSourcePId',
          'hetu_tag_level_info__hetu_face_id@featureSourcePId',
          'hetu_tag_level_info__hetu_level_five@featureSourcePId',
          'hetu_tag_level_info__hetu_level_four@featureSourcePId',
          'hetu_tag_level_info__hetu_level_one@featureSourcePId',
          'hetu_tag_level_info__hetu_level_three@featureSourcePId',
          'hetu_tag_level_info__hetu_level_two@featureSourcePId',
          'hetu_tag_level_info__hetu_tag@featureSourcePId',
          'hetu_tag_level_info_v2__hetu_cluster_id@featureSourcePId',
          'hetu_tag_level_info_v2__hetu_face_id@featureSourcePId',
          'hetu_tag_level_info_v2__hetu_level_four@featureSourcePId',
          'hetu_tag_level_info_v2__hetu_level_one@featureSourcePId',
          'hetu_tag_level_info_v2__hetu_level_three@featureSourcePId',
          'hetu_tag_level_info_v2__hetu_level_two@featureSourcePId',
          'hetu_tag_level_info_v2__hetu_tag@featureSourcePId',
          'hetu_sim_cluster_id@featureSourcePId',
          'author__exp_stat__exp_click',
          'author__exp_stat__exp_follow',
          'author__exp_stat__exp_forward',
          'author__exp_stat__exp_like',
          'author__exp_stat__exp_long_view',
          'author__exp_stat__exp_realshow',
          'author__exp_stat__exp_short_view',
          'author__exp_stat__exp_watch_time',
          'hetu_tag_level_info_v2__hetu_level_five',
          'hetu_sim_cluster_id',
          "mmu_face_gender"
        ],
        common_attrs = [
          "explore_request_data_set_tags",
          "llsid"
        ],
        for_debug_request_only = True
      ) \
      .log_debug_info(
        skip = "{{return request_type ~= \"life\"}}",
        item_list_from_attr = "life_realshow_common_list",
        item_attrs = [attr["name"] if not isinstance(attr, str) else attr for attr in self.base_attr_config],
        for_debug_request_only = True
      ) \
      .set_attr_value(  # 为了方便内流代码开发，此处先填充一些attr 及默认值
        no_overwrite=True,
        item_attrs=[
          {
            "name": "is_explore_photo",
            "type": "int",
            "value": 0
          },
          {
            "name": "is_high_quality_explore_photo",
            "type": "int",
            "value": 0
          },
          {
            "name": "cascade_plvtr2",
            "type": "double",
            "value": 0.0
          },
          {
            "name": "cascade_pdtr",
            "type": "double",
            "value": 0.0
          },
          {
            "name": "merchant_author_in_living",
            "type": "int",
            "value": 0
          },
          {
            "name": "cascade_final_index",
            "type": "int",
            "value": 0
          },
          {
            "name": "is_marketing_compensation_photo",
            "type": "int",
            "value": 0
          }
        ]
      ) \
      .copy_item_meta_info(
        save_reason_to_attr = "reason"
      )

  def enrich_picture_attrs(self):
    self.flow\
      .transform_item_attr(  # 判断是否是图片
        mappings=[
          {
            "check_attr_name": "upload_type",
            "check_attr_type": "int",
            "output_attr_name": "is_picture",
            "output_attr_type": "int",
            "output_default_value": 0,
            "rules": [{
              "check_values": [7, 10, 11, 70],
              "output_value": 1
            }]
          }
        ]
      ) \
      .transform_item_attr(  # 判断是否是图片
        mappings=[
          {
            "check_attr_name": "duration_ms",
            "check_attr_type": "int",
            "output_attr_name": "is_picture",
            "output_attr_type": "int",
            "output_default_value": 0,
            "rules": [{
              "check_range": {
                "upper_bound": 101,  # 不包含
              },
              "output_value": 1
            }]
          }
        ]
    )

  @property
  def base_attr_config(self) -> list:
    attrs = [
      "author__id",
      "cuckoo_info__is_cuckoo_photo",
      "cuckoo_info__author_type",
      "content_safety_level_with_namespace__level_hot_online",
      "percent_punish__data",
      "percent_punish__enabel_percent",
      "duration_ms",
      "upload_type",
      "upload_time",
      "photo_status",
      "is_merchant",
      "magic_face_id",
      "kuaishan_id",
      "outer_material_id",
      "topk_audit_level",
      "topk_audit_tag",
      "audit_hot_high_tag_level",
      "explore_operation_c_review_level",
      "audit_b_second_tag",
      "audit_hot_high_subdivision_level",
      "is_jianguan_risk_photo",
      "auto_audit_black_exempt_level_v1",
      "is_tnu_extend_index_photo",
      "photo_id",
      "dup_cluster_id",
      "pic_and_selfdup_id",
      "mmu_content_ids_3",
      "mmu_content_ids_8",
      "mmu_content_ids_9",
      "mmu_content_ids_10",
      "mmu_content_ids_14",
      "mmu_content_ids_15",
      "mmu_content_ids_16",
      "mmu_content_ids_17",
      "mmu_content_ids_33",
      "explore_stat",
      "hot_trend_generalized_info",
      "hetu_tag_level_info",
      "hetu_tag_level_info_v2",
      "hetu_tag_level_info__hetu_cluster_id",
      "hetu_tag_level_info__hetu_level_one",
      "hetu_tag_level_info__hetu_level_two",
      "hetu_tag_level_info__hetu_level_three",
      "hetu_tag_level_info__hetu_level_five",
      "hetu_tag_level_info__hetu_level_four",
      "hetu_tag_level_info__hetu_tag",
      "hetu_tag_level_info__hetu_face_id",
      "explore_stat__show_count",
      "explore_stat__real_show_count",
      "explore_stat__report_count",
      "nebula_stats__real_show_count",
      "nebula_stats__like_count",
      "nebula_stats__comment_count",
      "nebula_stats__forward_count",
      "nebula_stats__follow_count",
      "thanos_stats__real_show_count",
      "thanos_stats__like_count",
      "thanos_stats__comment_count",
      "thanos_stats__forward_count",
      "thanos_stats__follow_count",
      "fountain_stats__real_show_count",
      "fountain_stats__like_count",
      "fountain_stats__comment_count",
      "fountain_stats__forward_count",
      "fountain_stats__follow_count",
      "fountain_stats__report_count",
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
      "explore_stat__report_detail__total_report_count",
      "explore_stat__external_download",
      "report_detail__low_report_count",
      "report_detail__total_report_count",
      "live_photo_info__is_living",
      "mod",
      "tag",
      "photo_dnn_cluster_id",
      "author_high_score_v2",
      "mmu_img_cluster_v3",
      "mmu_img_cluster_v1",
      "mmu_img_cluster_v4",
      "mmu_cluster_music_id",
      "mmu_content_id",
      "ocr_cover_text_word_count",
      "music",
      "show_level_a",
      "show_level_b",
      "view_length_sum",
      "author__fans_count",
      "author__upload_count",
      "author__gender",
      "author_age_info__age_segment",
      "music_info__music_combo_id",
      "location__province_id",
      "location__city_id",
      "location__poi",
      "click_count",
      "like_count",
      "follow_count",
      "forward_count",
      "report_count",
      "click_upload_rate",
      "location",
      "author",
      "content_safety_level_with_namespace",
      "author_age_info",
      "lda_topic",
      "photo_high_end_status_bits",
      "long_term_photo",
      "user_hash_tag_id",
      "color",
      "comment_count",
      "chn",
      "height",
      "width",
      "filter",
      "infer_gender",
      "infer_year",
      "infer_gender_iter1",
      "infer_year_iter1",
      "location__community_type",
      "mmu_face_age",
      "mmu_face_gender",
      "mmu_photo_low_quality_model",
      "hetu_tag_level_info_v2__hetu_level_one",
      "hetu_tag_level_info_v2__hetu_level_two",
      "hetu_tag_level_info_v2__hetu_level_three",
      "hetu_tag_level_info_v2__hetu_tag",
      "hetu_tag_level_info_v3__hetu_level_one",
      "review_pass_level_b",
      "explore_stat__report_detail__low_report_count",
      "author__explore_report_thirtyday__low_report_count",
      "sim_remove_dup_id",
      "author__explore_report_thirtyday__total_report_count",
      "title_evil_level",
      "ocr_cover_text_evil_level",
      "audit_hot_cover_level",
      "merchant_item_info__item_id_list",
      "merchant_photo_cart_relation",
      "caption_length",
      "picture_type",
      "risk_level",
      "shuffle_policy",
      "nearby_feeling",
      "photo_picture_count",
      "mmu_low_quality_model_score_40",
      "mmu_low_quality_model_score_42",
      "mmu_low_quality_model_score_46",
      "mmu_low_quality_model_score_52",
      "mmu_low_quality_model_score_63",
      "mmu_low_quality_model_score_64",
      "mmu_low_quality_model_score_90",
      "mmu_low_quality_model_score_104",
      "mmu_low_quality_model_score_123",
      "mmu_low_quality_model_score_143",
      "mmu_low_quality_model_score_145",
      "mmu_low_quality_model_score_150",
      "mmu_low_quality_model_score_151",
      "mmu_low_quality_model_score_163",
      "mmu_low_quality_model_score_164",
      "is_sirius_punish",
      "is_support_author",
      "is_big_v_white_author_photo",
      "is_hot_rank_photo",
      "data_set_tags",
      "collect_count",
      "cold_start_breakout_score",
      "enable_download",
      "questionnaire_info__exposure_count",
      "questionnaire_info__negative_count",
      "questionnaire_info__positive_count",
      "questionnaire_info__unsure_count",
      "explore_questionnaire_info__exposure_count",
      "explore_questionnaire_info__negative_count",
      "explore_questionnaire_info__positive_count",
      "explore_questionnaire_info__unsure_count",
      {"name": "explore_punish_city", "type": "int_list"},
      "explore_punish",
      "high_hot_audit_tag_v2",
      "video_quality_assessment_flag",
      "eyeshot_source",
      "audit_user_experiment_level",
      "young_inc_tags",
      "final_cross_section_first_class_id",
      "light_inc_photo_flag",
      "author__hetu_author_tag__hetu_level_one",
      "author__hetu_author_tag__hetu_level_two",
      "author__hetu_author_tag__hetu_level_three",
      "high_value_pic_flag",
      "audit_cold_review_level",
      "audit_risk_immd_tag",
      "video_cold_start_info__photo_dynamic_xtrs_str",
      "photo_category_info__ecom_intent_score",
      "data_set_tags_bit",
      'fountain_stats__negative_count',
      'thanos_stats__negative_count',
      'nebula_stats__negative_count',
      'hetu_tag_level_info_v2__hetu_level_four',
      'hetu_tag_level_info_v2__hetu_face_id',
      'hetu_tag_level_info_v2__hetu_cluster_id',
      'online_lda_topic__ids',
      'author__category_detail__first_level_id',
      'author__category_detail__second_level_id',
      'author__category_detail__third_level_id',
      'author__category_detail__fourth_level_id',
      'GE_cluster_id',
      'mmu_text_cluster',
      'mmu_text_lda_topic',
      'is_mid_video_photo',
      'nearby_feeling__biz_name',
      'nearby_feeling__poi_info__poi_city_name',
      'nearby_feeling__poi_info__poi_district_name',
      'hetu_sim_cluster_id',
      "is_hotfire_yellow",
      "sirius_distribution_info__mark_cod",
      "author_grade_key",
      "author_shop_score",
      "author_max_item_score",
      "nebula_stats__collect_count",
      "thanos_stats__collect_count",
      "fountain_stats__collect_count",
      "explore_stat__collect_count",
      "secure_grading_action_code",
      "explore_stat__report_count",
      "fountain_stats__report_count",
      "thanos_stats__report_count",
      "nebula_stats__report_count",
      "explore_stat__short_play_count",
      "fountain_stats__short_play_count",
      "thanos_stats__short_play_count",
      "nebula_stats__short_play_count",
      "nebula_stats__view_length_sum",
      "thanos_stats__view_length_sum",
      "fountain_stats__view_length_sum",
    ]
    return attrs

  @property
  def hetu_v3_attr_config(self) -> list:
    base_attrs = self.base_attr_config
    attrs = [{"name": attr.replace("hetu_tag_level_info_v2", "hetu_tag_level_info_v3"), "as": attr} if isinstance(attr, str) and attr.startswith("hetu_tag_level_info_v2") and attr != "hetu_tag_level_info_v2" else attr for attr in base_attrs]
    return attrs

  @property
  def source_photo_base_attrs_splash(self):
    return [
      "hetu_tag_level_info_v2__hetu_level_one",
      "hetu_tag_level_info_v2__hetu_level_two",
      "hetu_tag_level_info_v2__hetu_level_three",
      "hetu_tag_level_info_v2__hetu_level_four",
      "hetu_tag_level_info_v2__hetu_tag",
      "hetu_tag_level_info_v2__hetu_face_id",
      "hetu_tag_level_info_v2__hetu_cluster_id",
      "hetu_tag_level_info__hetu_tag",
      "hetu_tag_level_info__hetu_level_one",
      "hetu_tag_level_info__hetu_level_two",
      "hetu_tag_level_info__hetu_level_three",
      "hetu_tag_level_info__hetu_level_four",
      "hetu_tag_level_info__hetu_level_five",
      "hetu_tag_level_info__hetu_face_id",
      "hetu_tag_level_info__hetu_cluster_id",
      "author__category_detail__third_level_id",
      "data_set_tags",
      "hetu_sim_cluster_id"
    ]

  @property
  def hetu_v3_source_photo_base_attrs_splash(self):
    attrs = [{"name": attr.replace("hetu_tag_level_info_v2", "hetu_tag_level_info_v3"), "as": attr} if isinstance(attr, str) and attr.startswith(
      "hetu_tag_level_info_v2") and attr != "hetu_tag_level_info_v2" else attr for attr in self.source_photo_base_attrs_splash]
    return attrs

  @property
  def photo_base_attrs_fast(self):
    photo_base_attrs_fast = [
      "photo_id",
      "author__id",
      "upload_type",
      "upload_time",
      "content_safety_level_with_namespace__level_hot_online",
      "explore_stat__click_count",
      "explore_stat__show_count",
      "explore_stat__comment_count",
      "explore_stat__negative_count",
      "explore_stat__follow_count",
      "explore_stat__forward_count",
      "explore_stat__view_length_sum",
      "explore_stat__report_detail__total_report_count",
      "explore_stat__real_show_count",
      "explore_stat__like_count",
      "explore_stat__report_count",
      "fountain_stats__negative_count",
      "fountain_stats__real_show_count",
      "fountain_stats__like_count",
      "fountain_stats__report_count",
      "thanos_stats__real_show_count",
      "nebula_stats__real_show_count",
      "thanos_stats__negative_count",
      "nebula_stats__negative_count",
      "thanos_stats__like_count",
      "nebula_stats__like_count",
      "explore_stat__like_count",
      "explore_stat__follow_count",
      "explore_stat__forward_count",
      "explore_stat__long_play_count",
      "explore_stat__real_show_count",
      "explore_stat__short_play_count",
      "author__exp_stat__exp_click",
      "author__exp_stat__exp_like",
      "author__exp_stat__exp_follow",
      "author__exp_stat__exp_long_view",
      "author__exp_stat__exp_realshow",
      "author__exp_stat__exp_forward",
      "author__exp_stat__exp_short_view",
      "author__exp_stat__exp_watch_time",
      "author__category_detail__first_level_id",
      "author__category_detail__second_level_id",
      "author__upload_count",
      "author__gender",
      "author_age_info__age_segment",
      "location__province_id",
      "location__city_id",
      "music_info__music_combo_id",
      "music",
      "tag",
      "mmu_img_cluster_v3",
      "view_length_sum",
      "click_count",
      "dup_cluster_id",
      "audit_hot_high_tag_level",
      "topk_audit_level",
      "topk_audit_tag",
      "duration_ms",
      "author__fans_count",
      "hetu_tag_level_info__hetu_level_one",
      "hetu_tag_level_info__hetu_level_two",
      "hetu_tag_level_info__hetu_level_three",
      "hetu_tag_level_info__hetu_level_five",
      "hetu_tag_level_info__hetu_tag",
      "hetu_tag_level_info__hetu_face_id",
      "live_photo_info__is_living",
      "hetu_tag_level_info_v2__hetu_level_two",
      "hetu_tag_level_info_v2__hetu_level_one",
      "hetu_tag_level_info_v2__hetu_level_three",
      "hetu_tag_level_info_v2__hetu_level_four",
      "hetu_tag_level_info_v2__hetu_level_five",
      "hetu_tag_level_info_v2__hetu_tag",
      "hetu_tag_level_info_v2__hetu_face_id",
      "hetu_tag_level_info_v2__hetu_cluster_id",
      "hetu_tag_level_info_v3__hetu_level_one",
      "explore_operation_c_review_level",
      "sim_remove_dup_id",
      "pic_and_selfdup_id",
      "is_merchant",
      "audit_b_second_tag",
      "audit_hot_high_subdivision_level",
      "photo_status",
      "is_mid_video_photo",
      "risk_level",
      "questionnaire_info__exposure_count",
      "questionnaire_info__positive_count",
      "questionnaire_info__negative_count",
      "questionnaire_info__unsure_count",
      "explore_questionnaire_info__exposure_count",
      "explore_questionnaire_info__negative_count",
      "explore_questionnaire_info__positive_count",
      "explore_questionnaire_info__unsure_count",
      "picture_type",
      "photo_picture_count",
      "long_term_photo",
      "mmu_content_ids_3",
      "mmu_content_ids_8",
      "mmu_content_ids_9",
      "mmu_content_ids_10",
      "mmu_content_ids_14",
      "mmu_content_ids_15",
      "mmu_content_ids_16",
      "mmu_content_ids_17",
      {"name": "explore_punish_city", "type": "int_list"},
      "data_set_tags",
      "height",
      "width",
      "high_hot_audit_tag_v2",
      "merchant_item_info__item_id_list",
      "merchant_photo_cart_relation",
      "eyeshot_source",
      "audit_user_experiment_level",
      "young_inc_tags",
      "final_cross_section_first_class_id",
      "light_inc_photo_flag",
      "high_value_pic_flag",
      "video_cold_start_info__photo_dynamic_xtrs_str",
      "audit_risk_immd_tag",
      "data_set_tags_bit",
      "hetu_sim_cluster_id",
      "is_hotfire_yellow",
      "sirius_distribution_info__mark_cod",
      "author_grade_key",
      "author_shop_score",
      "author_max_item_score",
      "nebula_stats__collect_count",
      "thanos_stats__collect_count",
      "fountain_stats__collect_count",
      "explore_stat__collect_count",
      "secure_grading_action_code",
      "explore_stat__report_count",
      "fountain_stats__report_count",
      "thanos_stats__report_count",
      "nebula_stats__report_count",
      "explore_stat__short_play_count",
      "fountain_stats__short_play_count",
      "thanos_stats__short_play_count",
      "nebula_stats__short_play_count",
      "nebula_stats__view_length_sum",
      "thanos_stats__view_length_sum",
      "fountain_stats__view_length_sum",
    ]
    return photo_base_attrs_fast

  @property
  def hetu_v3_photo_base_attrs_fast(self):
    photo_base_attrs_fast = [
      "photo_id",
      "author__id",
      "upload_type",
      "upload_time",
      "content_safety_level_with_namespace__level_hot_online",
      "explore_stat__click_count",
      "explore_stat__show_count",
      "explore_stat__comment_count",
      "explore_stat__negative_count",
      "explore_stat__follow_count",
      "explore_stat__forward_count",
      "explore_stat__view_length_sum",
      "explore_stat__report_detail__total_report_count",
      "explore_stat__real_show_count",
      "explore_stat__like_count",
      "fountain_stats__negative_count",
      "fountain_stats__real_show_count",
      "fountain_stats__like_count",
      "thanos_stats__real_show_count",
      "nebula_stats__real_show_count",
      "thanos_stats__negative_count",
      "nebula_stats__negative_count",
      "thanos_stats__like_count",
      "nebula_stats__like_count",
      "explore_stat__like_count",
      "explore_stat__follow_count",
      "explore_stat__forward_count",
      "explore_stat__long_play_count",
      "explore_stat__real_show_count",
      "explore_stat__short_play_count",
      "author__exp_stat__exp_click",
      "author__exp_stat__exp_like",
      "author__exp_stat__exp_follow",
      "author__exp_stat__exp_long_view",
      "author__exp_stat__exp_realshow",
      "author__exp_stat__exp_forward",
      "author__exp_stat__exp_short_view",
      "author__exp_stat__exp_watch_time",
      "author__category_detail__first_level_id",
      "author__category_detail__second_level_id",
      "author__upload_count",
      "author__gender",
      "author_age_info__age_segment",
      "location__province_id",
      "location__city_id",
      "music_info__music_combo_id",
      "music",
      "tag",
      "mmu_img_cluster_v3",
      "view_length_sum",
      "click_count",
      "dup_cluster_id",
      "audit_hot_high_tag_level",
      "topk_audit_level",
      "topk_audit_tag",
      "duration_ms",
      "author__fans_count",
      "hetu_tag_level_info__hetu_level_one",
      "hetu_tag_level_info__hetu_level_two",
      "hetu_tag_level_info__hetu_level_three",
      "hetu_tag_level_info__hetu_level_five",
      "hetu_tag_level_info__hetu_tag",
      "hetu_tag_level_info__hetu_face_id",
      "live_photo_info__is_living",
      "hetu_tag_level_info_v2__hetu_level_two",
      "hetu_tag_level_info_v2__hetu_level_one",
      "hetu_tag_level_info_v2__hetu_level_three",
      "hetu_tag_level_info_v2__hetu_level_four",
      "hetu_tag_level_info_v2__hetu_level_five",
      "hetu_tag_level_info_v2__hetu_tag",
      "hetu_tag_level_info_v2__hetu_face_id",
      "hetu_tag_level_info_v2__hetu_cluster_id",
      "hetu_tag_level_info_v3__hetu_level_one",
      "explore_operation_c_review_level",
      "sim_remove_dup_id",
      "pic_and_selfdup_id",
      "is_merchant",
      "audit_b_second_tag",
      "audit_hot_high_subdivision_level",
      "photo_status",
      "is_mid_video_photo",
      "risk_level",
      "questionnaire_info__exposure_count",
      "questionnaire_info__positive_count",
      "questionnaire_info__negative_count",
      "questionnaire_info__unsure_count",
      "explore_questionnaire_info__exposure_count",
      "explore_questionnaire_info__negative_count",
      "explore_questionnaire_info__positive_count",
      "explore_questionnaire_info__unsure_count",
      "picture_type",
      "photo_picture_count",
      "long_term_photo",
      "mmu_content_ids_3",
      "mmu_content_ids_8",
      "mmu_content_ids_9",
      "mmu_content_ids_10",
      "mmu_content_ids_14",
      "mmu_content_ids_15",
      "mmu_content_ids_16",
      "mmu_content_ids_17",
      {"name": "explore_punish_city", "type": "int_list"},
      "data_set_tags",
      "height",
      "width",
      "high_hot_audit_tag_v2",
      "merchant_item_info__item_id_list",
      "merchant_photo_cart_relation",
      "eyeshot_source",
      "audit_user_experiment_level",
      "young_inc_tags",
      "final_cross_section_first_class_id",
      "light_inc_photo_flag",
      "high_value_pic_flag",
      "video_cold_start_info__photo_dynamic_xtrs_str",
      "audit_risk_immd_tag",
      "data_set_tags_bit",
      "hetu_sim_cluster_id",
      "is_hotfire_yellow",
      "sirius_distribution_info__mark_cod",
      "author_grade_key",
      "author_shop_score",
      "author_max_item_score",
      "nebula_stats__collect_count",
      "thanos_stats__collect_count",
      "fountain_stats__collect_count",
      "explore_stat__collect_count",
      "secure_grading_action_code",
      "explore_stat__report_count",
      "fountain_stats__report_count",
      "thanos_stats__report_count",
      "nebula_stats__report_count",
      "explore_stat__short_play_count",
      "fountain_stats__short_play_count",
      "thanos_stats__short_play_count",
      "nebula_stats__short_play_count",
      "nebula_stats__view_length_sum",
      "thanos_stats__view_length_sum",
      "fountain_stats__view_length_sum",
    ]
    hetu_v3_attrs = [{"name": attr.replace("hetu_tag_level_info_v2", "hetu_tag_level_info_v3"), "as": attr} if isinstance(attr, str) and attr.startswith(
      "hetu_tag_level_info_v2") and attr != "hetu_tag_level_info_v2" else attr for attr in photo_base_attrs_fast]
    return hetu_v3_attrs


