from retrieval import CommonModule

class FetchColossusRetrievalTriggerModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self):
    self.flow \
      .if_("enable_target_item_hetu_set > 0", to_be_delete = "date=2023-11-16;committer=liyunhao") \
        .copy_attr(
          attrs = [{
            "from_common": "target_item_hetu_set_exp",
            "to_common": "knowledge_hetu_set"
          }]
        ) \
      .else_() \
        .if_("enable_knowledge_new_hetu_set > 0") \
          .copy_attr(
            attrs = [{
              "from_common": "knowledge_hetu_set_exp",
              "to_common": "knowledge_hetu_set"
            }]
          ) \
        .else_() \
          .copy_attr(
            attrs = [{
              "from_common": "knowledge_hetu_set_base",
              "to_common": "knowledge_hetu_set"
            }]
          ) \
        .end_() \
      .end_() \
      .if_("enable_colossus_trigger_request_redis == 1") \
        .retrieve_by_redis(
          cluster_name = "recoColossusTriggers",
          timeout_ms = 5,
          reason = 0,
          key_from_attr = "_USER_ID_",
          key_prefix = "{{redis_key_prefix}}",
          retrieve_num = 3000,
          item_separator = ",",
          attr_separator = "@",
          #item_regex = r"[^|,](\d+)@(\d+)@",
          extra_item_attrs = [
            {"name": "trigger_play_time", "type": "int"},
            {"name": "trigger_author_id", "type": "int"},
            {"name": "trigger_tag", "type": "int"},
            {"name": "trigger_duration", "type": "int"},
          ],
          save_result_to_common_attr = "colossus_user_info__trigger_id_list",
        ) \
      .end_() \
      .if_("colossus_user_info__trigger_id_list ~= nil and #colossus_user_info__trigger_id_list > 0") \
        .pack_item_attr(
          item_source = {
            "common_attr": ["colossus_user_info__trigger_id_list"]
          },
          mappings = [
            {
              "from_item_attr": "trigger_play_time",
              "aggregator": "concat",
              "to_common_attr": "colossus_user_info__trigger_weight_list",
              "default_val": 1
            },
            {
              "from_item_attr": "trigger_tag",
              "aggregator": "concat",
              "to_common_attr": "colossus_user_info__trigger_tag_list",
              "default_val": 0
            },
            {
              "from_item_attr": "trigger_author_id",
              "aggregator": "concat",
              "to_common_attr": "colossus_user_info__trigger_author_list",
              "default_val": 0
            },
          ]
        ) \
        .if_("enable_knowledge_trigger > 0") \
          .enrich_attr_by_lua(
            import_common_attr = ["colossus_user_info__trigger_tag_list", "colossus_user_info__trigger_id_list", "colossus_user_info__trigger_weight_list", "knowledge_hetu_set", "knowledge_trigger_play_time_ths"],
            export_common_attr = ["colossus_user_info__knowledge_trigger_set"],
            function_for_common = "extract_knowledge_trigger",
            lua_script_file = "explore/retrieval/lua/module/colossus_user_info__fetch_knowledge_trigger.lua"
          ) \
        .end_() \
      .else_() \
        .explore_colossus_v2_trigger_enrich(
          colossus_resp_attr = "colossus_resp_v2",
          output_colossus_trigger_attr = "colossus_user_info__trigger_id_list",
          output_colossus_trigger_weight_attr = "colossus_user_info__trigger_weight_list",
          output_colossus_trigger_author_attr = "colossus_user_info__trigger_author_list",
          output_colossus_trigger_tag_attr = "colossus_user_info__trigger_tag_list",
          output_knowledge_trigger_attr = "colossus_user_info__knowledge_trigger_set",
          output_interest_explore_trigger_attr = "colossus_user_info__interest_explore_trigger_set",
          output_colossus_info_attr = "redis_val",
          enable_progressive_trigger = "{{enable_progress_trigger}}",
          min_days_ago = "{{min_days_ago}}",
          max_days_ago = "{{max_days_ago}}",
          trigger_select_num = "{{trigger_select_num}}",
          trigger_sample_amplifier = "{{trigger_sample_amplifier}}",
          trigger_min_play_time = "{{play_time_ths}}",
          trigger_select_alpha = "{{progressive_alpha}}",
          trigger_select_base_num = "{{progressive_base}}",
          trigger_select_topk = "{{progressive_topk}}",
          trigger_select_skip_num = "{{progressive_skip_num}}",
          enable_knowledge_trigger = "{{enable_knowledge_trigger}}",
          knowledge_trigger_max_num = "{{knowledge_trigger_limit}}",
          knowledge_trigger_play_time_ths = "{{knowledge_trigger_play_time_ths}}",
          enable_completion_trigger = "{{enable_completion_trigger}}",
          completion_trigger_num = "{{completion_trigger_num}}",
          knowledge_hetu_set_attr = "knowledge_hetu_set",
          enable_interest_explore_trigger = "{{enable_interest_explore_trigger}}",
          interest_explore_trigger_play_time_ths = "{{interest_explore_trigger_play_time_ths}}",
          interest_explore_hetu_set_attr = "colossus_explore_hetu_tags",
          enable_cluster_trigger = "{{enable_cluster_trigger}}",
          cluster_trigger_preserve_ratio = "{{cluster_trigger_preserve_ratio}}",
          trigger_select_interval_size = "{{trigger_select_interval_size}}",
          cluster_trigger_play_time_ths = "{{cluster_trigger_play_time_ths}}"
        ) \
        .copy_user_meta_info(
          save_user_id_to_attr = "user_id"
        ) \
        .write_to_redis(
          kcc_cluster = "recoColossusTriggers",
          timeout = 10,
          key_prefix = "{{redis_key_prefix}}",
          expire_second = "{{redis_expire_seconds}}",
          key = "{{user_id}}",
          value = "{{redis_val}}"
        ) \
      .end_()

  def post_process(self) -> None:
    self.flow \
      .log_debug_info(
        item_attrs = [
          "trigger_duration@colossus_user_info__trigger_id_list",
        ],
        item_num_limit = 0,
      )
