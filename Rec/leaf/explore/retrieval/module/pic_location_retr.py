from retrieval.retrieval_module import RetrievalModule


class PicLocationRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    """
    default : u2i
    case 1: u2i, 远程拉取user emb, 扩大user emb 的命中率
    case 2: i2i
    case 3: delegate_retrieve 方便处理复杂逻辑
    """
    self.flow\
    .switch_("retr_mode")\
      .case_(1) \
        .get_remote_embedding_lite(
          kess_service="{{emb_server_name}}",
          shard_num=1,
          timeout_ms=20,
          id_converter={
            "type_name": "kuibaEmbeddingIdConverter"
          },
          size=128,
          slot=0,
          output_attr_name="user_emb",
          query_source_type="user_id",
          client_side_shard=True
        ) \
        .retrieve_by_ann_embedding(
          reason=self.reason,
          kess_service="{{retr_service}}",
          space="cosine",
          timeout_ms=50,
          items_from_attr=["_USER_ID_"],
          embeddings_from_attr=["user_emb"],
          bound_type={
            "total_limit": "{{retr_total_num}}",
          },
          algo_type={
            "scann": {},
          },
          src_bucket="user",
          dest_bucket="{{dest_bucket}}"
        )\
      .case_(2) \
        .explore_colossus_v2_pic_trigger_enrich(
          colossus_resp_attr="colossus_resp_v2",
          output_colossus_trigger_attr="colossus_pic_trigger_list",
          output_colossus_trigger_tag_attr="colossus_pic_trigger_tag_list",
          picture_trigger_num="{{picture_trigger_num}}",
          picture_trigger_interact_num="{{picture_trigger_interact_num}}",
          colossus_channel_select_str="{{colossus_channel_select_str}}",
          enable_exclude_single_pic="{{enable_exclude_single_pic}}",
          seleted_eff_play_thd_sec="{{seleted_eff_play_thd_sec}}",
          colossus_range_days="{{colossus_range_days}}",
        ) \
        .pack_common_attr(
          input_common_attrs=[
            "global_normal_trigger_list",
            "global_high_value_trigger_list",
            "global_normal_trigger_weight_list",
            "global_high_value_trigger_weight_list",
            "explore_selected_trigger_list"
          ],
          output_common_attr="trigger_video_ids",
          deduplicate=True
        )\
        .shuffle_list_attr(
          common_attr= "trigger_video_ids"
        ) \
        .truncate(
          item_list_from_attr="trigger_video_ids",
          size_limit="{{trigger_video_size}}",
        ) \
        .shuffle_list_attr(
          common_attr="colossus_pic_trigger_list"
        ) \
        .truncate(
          item_list_from_attr="colossus_pic_trigger_list",
          size_limit="{{trigger_pic_size}}",
        ) \
        .pack_common_attr(
          input_common_attrs=[
            "colossus_pic_trigger_list",
            "trigger_video_ids"
          ],
          output_common_attr="trigger_photo_ids",
          deduplicate=True
        ) \
        .retrieve_by_ann_embedding(
          reason=self.reason,
          kess_service="{{retr_service}}",
          space="cosine",
          timeout_ms=50,
          items_from_attr=["trigger_photo_ids"],
          bound_type={
            "total_limit": "{{retr_total_num}}",
          },
          algo_type={
            "scann": {},
          },
          src_bucket="photo",
          dest_bucket="{{dest_bucket}}"
        )\
      .case_(3) \
        .copy_user_meta_info(
          save_request_type_to_attr="request_type",
        )\
        .delegate_retrieve(
          reason=self.reason,
          kess_service="{{retr_service}}",
          timeout_ms=50,
          request_type="{{request_type}}",
          request_num="{{retr_total_num}}",
          send_common_attrs_in_request=False,
          send_common_attrs=[
            {"name": "userInfo", "as": "user"},
          ],
          reset_item_type=0
        )\
      .default_() \
        .retrieve_by_ann_embedding(
          reason=self.reason,
          kess_service="{{retr_service}}",
          space="cosine",
          timeout_ms=50,
          items_from_attr=["_USER_ID_"],
          bound_type={
            "total_limit": "{{retr_total_num}}",
          },
          algo_type={
            "scann": {},
          },
          src_data_type="user",
          src_bucket="user",
          dest_bucket="{{dest_bucket}}"
        )\
    .end_()\
    .deduplicate()\
    .filter_by_common_attr(
      common_attr=["browse_screen__pid_list"]
    )
