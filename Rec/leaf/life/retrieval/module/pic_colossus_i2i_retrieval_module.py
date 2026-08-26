from retrieval.retrieval_module import RetrievalModule

class PicColossusI2IRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .if_("trigger_hetu_white_list_str ~= nil and trigger_hetu_white_list_str ~= \"\"") \
        .split_string(
          input_common_attr = "trigger_hetu_white_list_str",
          output_common_attr = "hetu_trigger_white_list",
          delimiters = ",",
          parse_to_int = True,
        ) \
      .end_() \
      .explore_colossus_v2_pic_trigger_enrich(
        colossus_resp_attr = "colossus_resp_v2",
        output_colossus_trigger_attr = "colossus_trigger_list",
        output_colossus_trigger_tag_attr = "colossus_trigger_tag_list",
        picture_trigger_num = "{{colossus_pic_all_limit}}",
        picture_trigger_interact_num="{{colossus_pic_interact_limit}}",
        colossus_channel_select_str="{{colossus_pic_trigger_channel_str}}",
        enable_exclude_single_pic ="{{enable_exclude_single_pic}}",
        seleted_eff_play_thd_sec="{{seleted_eff_play_thd_sec}}",
        colossus_range_days="{{colossus_range_days}}",
        colossus_hetu_white_list = "{{hetu_trigger_white_list}}"
      ) \
      .if_("colossus_trigger_list == nil or #colossus_trigger_list <= 0") \
        .return_() \
      .end_() \
      .if_("enable_scatter_trigger > 0") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "colossus_trigger_list", "as": "trigger_list"},
            {"name": "colossus_trigger_tag_list", "as": "trigger_scatter_attr_list"},
            {"name": "hetu_low_level_tags", "as": "scatter_map_keys"},
            {"name": "hetu_level2_tags", "as": "scatter_map_values"},
            {"name": "colossus_trigger_limit", "as": "total_limit"},
            {"name": "scatter_each_limit", "as": "each_limit"}
          ],
          export_common_attr = [
            {"name": "final_trigger_list", "as": "colossus_trigger_list"}
          ],
          function_name = "ScatterTriggers",
          class_name = "ExploreLightFunctionSetV2"
        ) \
      .end_() \
      .shuffle_list_attr(common_attr="colossus_trigger_list") \
      .pack_common_attr(
        input_common_attrs = ["colossus_trigger_list"],
        output_common_attr = "colossus_trigger_list",
        limit_num = "{{colossus_trigger_limit}}",
      ) \
      .if_("use_colossus_trigger_embedding > 0") \
        .get_remote_embedding_lite(
          kess_service = "{{embedding_service_name}}",
          shard_num = 8,
          timeout_ms = 20,
          id_converter = {
          "type_name": "kuibaEmbeddingIdConverter"
          },
          size = 128,
          input_attr_name = "colossus_trigger_list",
          output_attr_name = "colossus_trigger_embedding",
          query_source_type = "common_attr",
          client_side_shard = True
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "colossus_trigger_list", "as": "trigger_list"},
            {"name": "colossus_trigger_embedding", "as": "trigger_embedding_list"},
          ],
          export_common_attr = [
            {"name": "trigger_list", "as": "colossus_trigger_list"},
            {"name": "trigger_embedding_list", "as": "colossus_trigger_embedding"},
          ],
          function_name = "GetValidEmbeddings",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .retrieve_by_ann_embedding(
          kess_service = "{{ann_service_name}}",
          space = "cosine",
          timeout_ms = 50,
          reason = self.reason,
          shard_num = 1,
          items_from_attr = ["colossus_trigger_list"],
          embeddings_from_attr = ["colossus_trigger_embedding"],
          bound_type = {
            "top_k": "{{ann_top_k}}",
          },
          algo_type = {
            "scann": {},
          },
          src_bucket = "{{src_bucket}}",
          dest_bucket = "{{dest_bucket}}",
        ) \
      .else_() \
        .retrieve_by_ann_embedding(
          kess_service = "{{ann_service_name}}",
          space = "cosine",
          timeout_ms = 50,
          reason = self.reason,
          shard_num = 1,
          items_from_attr = ["colossus_trigger_list"],
          bound_type = {
            "top_k": "{{ann_top_k}}",
          },
          algo_type = {
            "scann": {},
          },
          src_bucket = "{{src_bucket}}",
          dest_bucket = "{{dest_bucket}}",
        ) \
      .end_() \
      .deduplicate() \
      .filter_by_common_attr(
        common_attr = ["browse_screen__pid_list"]
      ) \
      .if_("enable_shuffle_result == 1") \
        .shuffle() \
      .end_() \
      .limit("{{retrieve_num}}")
          