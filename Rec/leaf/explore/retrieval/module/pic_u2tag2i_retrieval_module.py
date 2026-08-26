from retrieval import RetrievalModule

class PicU2Tag2IRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
    
  def process(self) -> None:
    self.flow \
      .if_("enable_explore_pic_u2tag2i_statistics_trigger == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "user_valid_tag_id_list", "as": "tag_id_list"},
            {"name": "user_valid_tag_score_list", "as": "tag_score_list"},
            {"name": "explore_pic_u2tag2i_tag_score_thresh", "as": "tag_score_thresh"},
            {"name": "explore_pic_u2tag2i_retr_trigger_num", "as": "tag_max_num"},
          ],
          export_common_attr = [
            {"name": "valid_tag_id_list", "as": "user_valid_tag_list"},
          ],
          function_name = "GetValidTagIdList",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("enable_explore_pic_u2tag2i_action_trigger == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "user_info_ptr",
            {"name": "pic_u2tag2i_play_cnt", "as": "play_cnt"},
            {"name": "pic_u2tag2i_like_cnt", "as": "like_cnt"},
            {"name": "pic_u2tag2i_follow_cnt", "as": "follow_cnt"},
            {"name": "pic_u2tag2i_comment_cnt", "as": "comment_cnt"},
            {"name": "pic_u2tag2i_collect_cnt", "as": "collect_cnt"},
            {"name": "pic_u2tag2i_forward_cnt", "as": "forward_cnt"},
            {"name": "pic_u2tag2i_download_cnt", "as": "download_cnt"},
            {"name": "explore_pic_u2tag2i_action_trigger_tag_max", "as": "tag_max"},
            {"name": "explore_pic_u2tag2i_action_trigger_tag_min", "as": "tag_min"},
            {"name": "explore_pic_u2tag2i_action_trigger_tag_num", "as": "tag_max_num"}
          ],
          export_common_attr = [
            {"name": "action_trigger_tag_list", "as": "action_trigger_tag_list"},
          ],
          function_name = "GetActionListTagTrigger",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .pack_common_attr(
          input_common_attrs = [
            "user_valid_tag_list",
            "action_trigger_tag_list"
          ],
          output_common_attr = "user_valid_tag_list",
          deduplicate = True,
        ) \
      .end_() \
      .if_("user_valid_tag_list == nil or #user_valid_tag_list == 0") \
        .return_() \
      .end_() \
      .retrieve_by_redis(
        reason = self.reason,
        retrieve_num = "{{explore_pic_u2tag2i_retr_num}}",
        retrieve_num_per_key = "{{explore_pic_u2tag2i_retr_num_per_key}}",
        cluster_name = "explorePicCache",
        timeout_ms = 50,
        key_from_attr = "user_valid_tag_list",
        key_prefix = "{{explore_pic_u2tag2i_redis_key_prefix}}",
        item_separator = ","
      ) \
      .deduplicate() \
      .shuffle() \
      .limit(size = "{{explore_pic_u2tag2i_retr_num_final}}")

