from filter import CommonModule
from . import common_filter

class FilterDefaultModule(CommonModule):
  ITEM_ATTR_MAP = {
  }

  FILTERS = [
    {
      "name": "follow_author",
      "enable": "{{fountain_enable_follow_author_filter}}",
      "follow_author_filter_timegap_attr": "fountain_follow_author_filter_timegap",
      "author_id_attr": "author__id",
      "upload_time_attr": "upload_time",
    },
  ]

  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self):
    common_filter.append_prepare_processors(self.flow)
    self.flow \
      .if_("fountain_enable_fast_retr_filter_limit == 1") \
        .explore_retrieval_filter(
          name = "explore_retr_filter_fast_limit",
          traceback = True,
          user_info_ptr_attr = "userInfoPb",
          item_attr_map = {**common_filter.ITEM_ATTR_MAP, **FilterDefaultModule.ITEM_ATTR_MAP},
          filters = common_filter.FILTERS + FilterDefaultModule.FILTERS,
        ) \
      .else_() \
        .explore_retrieval_filter(
          name = "explore_retr_filter_fast",
          traceback = True,
          user_info_ptr_attr = "userInfoPb",
          item_attr_map = {**common_filter.ITEM_ATTR_MAP, **FilterDefaultModule.ITEM_ATTR_MAP},
          filters = common_filter.FILTERS + FilterDefaultModule.FILTERS,
          truncation_map = {
            "default": 5000,
          },
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
        check_point = "filter_finish"
      )
