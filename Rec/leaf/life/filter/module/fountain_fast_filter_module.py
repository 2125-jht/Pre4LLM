from filter import CommonModule


class FountainFastFilterModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  @property
  def item_attr_map(self) -> dict:
    attr_map = {
      "explore_server_show_attr": "explore_stat__show_count",
    }
    attr_map.update(self.flow.base_item_attr_map)
    return attr_map

  @property
  def filters(self) -> list:
    filter_list = [
    ]
    filter_list += self.flow.base_filters
    return filter_list

  @property
  def sec_tab_truncation_map(self) -> dict:
    return {
    }

  @property
  def truncation_map(self) -> dict:
    return {
      "196": 360,
      "6300": 350,
    }

  def process(self):
    self.flow \
      .base_params() \
      .explore_life_retrieval_filter(
        user_info_ptr_attr="user_info_ptr",
        item_attr_map=self.item_attr_map,
        filters=self.filters,
        truncation_map={
          "default": 5000,
        },
      )
    self.cid_dedup()

  def cid_dedup(self):
    self.flow\
      .set_attr_value(
        # 后链路首屏非首屏共享，暂时hack attr满足依赖检测。
        no_overwrite=True,
        item_attrs=[
          {
            "name": "source_related_score",
            "type": "double",
            "value": 0
          }
        ]
      ) \
      .if_("xlife_fountain_skip_dedup_content_ids_in_pagesize == 0") \
        .split_string(
          input_common_attr = "fountain_hot_high_photo_skip_filter_types_str",
          output_common_attr = "fountain_hot_high_photo_skip_filter_types",
          delimiters = ",",
          parse_to_int = True,
        ) \
        .split_string(
          input_common_attr = "fountain_cid_browse_set_hetu_list_str",
          output_common_attr = "cid_browse_set_hetu_list",
          delimiters = ",",
          parse_to_int = True,
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
          cid_browse_set_hetu_list = "{{cid_browse_set_hetu_list}}"
        ) \
        .perflog_reason_count(
          check_point = "filter_by_dedup_content_ids_in_pagesize"
        ) \
      .end_() \

  def post_process(self) -> None:
    self.flow\
      .log_debug_info(
        common_attrs=[
          'rank_neg_photo_id_list_str',
          'rerank_neg_photo_id_list_str',
          'explore_nearline_user_update_flag',
        ],
        item_attrs=[
          'audit_cold_review_level',
          'audit_hot_high_subdivision_level',
          'audit_risk_immd_tag',
          'audit_user_experiment_level',
          'author__explore_report_thirtyday__low_report_count',
          'author__explore_report_thirtyday__total_report_count',
          'auto_audit_black_exempt_level_v1',
          'cuckoo_info__author_type',
          'data_set_tags_bit',
          'dup_cluster_id',
          'enable_download',
          'explore_operation_c_review_level',
          'explore_stat__report_detail__low_report_count',
          'final_cross_section_first_class_id',
          'fountain_stats__like_count',
          'fountain_stats__real_show_count',
          'hetu_tag_level_info_v2__hetu_tag',
          'hetu_tag_level_info_v3__hetu_level_one',
          'high_hot_audit_tag_v2',
          'is_jianguan_risk_photo',
          'is_sirius_punish',
          'kuaishan_id',
          'mmu_content_ids_10',
          'mmu_content_ids_14',
          'mmu_content_ids_15',
          'mmu_content_ids_16',
          'mmu_content_ids_17',
          'mmu_content_ids_3',
          'mmu_content_ids_8',
          'mmu_content_ids_9',
          'mmu_low_quality_model_score_104',
          'mmu_low_quality_model_score_123',
          'mmu_low_quality_model_score_143',
          'mmu_low_quality_model_score_145',
          'mmu_low_quality_model_score_150',
          'mmu_low_quality_model_score_151',
          'mmu_low_quality_model_score_163',
          'mmu_low_quality_model_score_164',
          'mmu_low_quality_model_score_40',
          'mmu_low_quality_model_score_42',
          'mmu_low_quality_model_score_46',
          'mmu_low_quality_model_score_52',
          'mmu_low_quality_model_score_63',
          'mmu_low_quality_model_score_64',
          'mmu_low_quality_model_score_90',
          'nebula_stats__like_count',
          'nebula_stats__real_show_count',
          'ocr_cover_text_evil_level',
          'outer_material_id',
          'photo_category_info__ecom_intent_score',
          'photo_status', 'pic_and_selfdup_id',
          'review_pass_level_b',
          'risk_level',
          'sim_remove_dup_id',
          'thanos_stats__like_count',
          'thanos_stats__real_show_count',
          'title_evil_level',
          'topk_audit_tag',
          'video_cold_start_info__photo_dynamic_xtrs_str',
          'video_quality_assessment_flag',
          'young_inc_tags',
        ],
        for_debug_request_only=True
      )
