from retrieval.retrieval_module import RetrievalModule

class PicSsmI2IRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    if self.copy_from_source_hetu_level_one:
      self.flow \
        .copy_attr(
          attrs = [{
            "from_common": "source_hetu_level_one",
            "to_common": "pic_trigger_white_list"
          }]
        )
    elif self.split_from_pic_hetu_white_list_str:
      self.flow \
        .split_string(
          input_common_attr = "fountain_pic_colossus_trigger_hetu_white_list_str",
          output_common_attr = "pic_trigger_white_list",
          delimiters = ",",
          parse_to_int = True,
        )
    
    self.flow \
      .explore_colossus_v2_pic_trigger_enrich(
        colossus_resp_attr = "colossus_resp_v2",
        output_colossus_trigger_attr = "colossus_trigger_list",
        picture_trigger_num = "{{fountain_pic_colossus_all_limit}}",
        picture_trigger_interact_num = "{{fountain_pic_colossus_interact_limit}}",
        colossus_channel_select_str = "{{fountain_pic_colossus_trigger_channel_str}}",
        enable_exclude_single_pic = "{{fountain_pic_colossus_exclude_single_pic}}",
        seleted_eff_play_thd_sec = "{{fountain_pic_colossus_eff_play_thd_sec}}",
        colossus_range_days = "{{fountain_pic_colossus_range_days}}",
        colossus_hetu_white_list = "{{pic_trigger_white_list}}"
      ) \
      .log_debug_info(
        common_attrs = [
          "fountain_pic_colossus_all_limit",
          "fountain_pic_colossus_interact_limit",
          "fountain_pic_colossus_trigger_channel_str",
          "fountain_pic_colossus_exclude_single_pic",
          "fountain_pic_colossus_eff_play_thd_sec",
          "fountain_pic_colossus_range_days",
          "source_hetu_level_one",
          "pic_trigger_white_list",
          "colossus_trigger_list",
        ],
        for_debug_request_only = True,
      ) \
      .if_("colossus_trigger_list == nil or #colossus_trigger_list <= 0") \
        .return_() \
      .end_() \
      .shuffle_list_attr(
        common_attr = "colossus_trigger_list") \
      .pack_common_attr(
        input_common_attrs = ["colossus_trigger_list"],
        output_common_attr = "colossus_trigger_list",
        limit_num = "{{fountain_pic_colossus_trigger_limit}}",
      ) \
      .retrieve_by_ann_embedding(
        kess_service = "{{fountain_pic_ssm_i2i_retr_kess_name}}",
        timeout_ms = 100,
        reason = self.reason,
        space = "cosine",
        items_from_attr = ["colossus_trigger_list"],
        attr_single_limit = 100,
        bound_type = {
          "top_k": "{{fountain_pic_ssm_i2i_retr_topk}}",
        },
        algo_type = {
          "faiss": {}
        },
        src_bucket = "{{fountain_pic_ssm_i2i_src_bucket}}",
        dest_bucket = "{{fountain_pic_ssm_i2i_dest_bucket}}",
        dest_bucket_item_type = 1
      ) \
      .filter_by_browse_set() \
      .truncate(
        size_limit = "{{fountain_pic_ssm_i2i_retr_total_num}}"
      )

  @property    
  def copy_from_source_hetu_level_one(self) -> str:
    return self.config.get("copy_from_source_hetu_level_one", False)
  
  @property
  def split_from_pic_hetu_white_list_str(self) -> str:
    return self.config.get("split_from_pic_hetu_white_list_str", False)