from retrieval import RetrievalModule
from dragonfly.common_leaf_dsl import LeafService, LeafFlow

class HatePhotoI2IRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  @classmethod
  def is_retrieval(cls) -> bool:
    return True
    
  def process(self) -> None:
    ## 从 global trigger 里完成抽取
   
    self.flow \
      .if_("enable_hate_list_trigger_retrieval == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "hate_list",
            "hate_list_timestamps"
          ],
          export_common_attr = [
            {"name": "hate_trigger_list", "as": "hate_pids"}
          ],
          function_name = "GenSwingNegI2IHateList",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("enable_short_view_trigger_retrieval == 1 and short_view_trigger_num > 0") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "short_view_trigger_num",
            "latest_play_minute",
            "short_view_play_time_ms_threshold",
            "user_info_ptr"
          ],
          export_common_attr = [
            {"name": "trigger_list", "as": "short_view_pids"}
          ],
          function_name = "TriggerFromShortView",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("enable_realshow_no_click_trigger_retrieval == 1 and realshow_no_click_trigger_num > 0") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "user_info_ptr",
            "latest_showtime_minute",
            "realshow_no_click_trigger_num",
            {"name": "realshow_no_click_trigger_num", "as": "keep_size"}
          ],
          export_common_attr = [
            "real_show_no_click_pids",
          ],
          function_name = "TriggerFromRecentRealShowNoClick",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .pack_common_attr(
        input_common_attrs = [
          "hate_pids", 
          "short_view_pids",
          "real_show_no_click_pids"
        ],
        output_common_attr = "seed_photos",
        deduplicate = True
      ) \
      .if_("seed_photos == nil or #seed_photos == 0") \
        .return_() \
      .end_() \
      .shuffle_list_attr(
        common_attr = "seed_photos"
      ) \
      .pack_common_attr(
        input_common_attrs = ["seed_photos"],
        output_common_attr = "seed_photos",
        limit_num = "{{trigger_num}}"
      ) \
      .if_("enable_emb_sever == 1") \
        .get_remote_embedding_lite_v2(
          protocol = 1,
          colossusdb_embd_service_name = "grpc_clsdb_ps-hate-embed",
          colossusdb_embd_table_name = "grpc_clsdb_ps-hate-embed",
          id_converter = {"type_name": "plainIdConverter"},
          slot = 0,
          input_attr_name = "seed_photos",
          output_attr_name = "seed_photo_embedding",
          query_source_type = "common_attr",
          raw_data_type = 'float32',
          colossusdb_use_kconf_client = False,
          size = 128,
          shard_num = 8
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "seed_photos", "as": "trigger_list"},
            {"name": "seed_photo_embedding", "as": "trigger_embedding_list"},
            {"name": "emb_size", "as": "dim"}
          ],
          export_common_attr = [
            {"name": "trigger_list", "as": "seed_photos"}, 
            {"name": "trigger_embedding_list", "as": "seed_photo_embedding"}
          ],
          function_name = "GetValidEmbeddings",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .retrieve_by_ann_embedding(
          reason = self.reason,
          kess_service = "{{ann_service_name}}",
          shard_num = 1,
          timeout_ms = 70,
          space = "cosine",
          items_from_attr = ["seed_photos"],
          embeddings_from_attr = ["seed_photo_embedding"],
          bound_type = {
            "top_k": "{{retrieve_num_per_trigger}}",
          },
          algo_type = {
            "scann": {},
          },
          src_bucket = "photo",
          dest_bucket = "photo",
          save_distance_to_attr = "ann_dist_list"
        ) \
      .else_() \
        .retrieve_by_ann_embedding(
          reason = self.reason,
          kess_service = "{{ann_service_name}}",
          shard_num = 1,
          timeout_ms = 70,
          space = "cosine",
          items_from_attr = ["seed_photos"],
          bound_type = {
            "top_k": "{{retrieve_num_per_trigger}}",
          },
          algo_type = {
            "scann": {},
          },
          src_bucket = "photo",
          dest_bucket = "photo",
          save_distance_to_attr = "ann_dist_list"
        ) \
      .end_() \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "ann_dist_threshold"
        ],
        import_item_attr = [
          "ann_dist_list"
        ],
        export_common_attr = [
          {"name": "remain_pid_list", "as": "hate_photo_i2i_neg_pid_list"}
        ],
        function_name = "GetPidListWithAnnDistThreshold",
        class_name = "ExploreLightFunctionSetV2"
      ) \
      .limit(size = 0)
    
  @property
  def emb_size(self) -> int:
    return self.config.get("emb_size", 128)