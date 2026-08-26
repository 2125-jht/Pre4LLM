from retrieval import RetrievalModule

class MmuSimEmbDupI2IRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
    
  def process(self) -> None:
    self.flow \
      .if_("request_type ~= 'life' and request_type ~= 'fountain_fast_v1_life' and request_type ~= 'fountain_fast_life_pic_inside'") \
        .return_() \
      .end_() \
      .if_("enable_new_version > 0") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "browse_trigger_num",
            "hate_trigger_num",
            "hate_list",
            {"name": "browse_screen__pid_list", "as": "browse_set"}
          ],
          export_common_attr = [
            {"name": "trigger_list", "as": "dup_seed_photos"}
          ],
          function_name = "TriggerFromBrowseSetAndHateList",
          class_name = "ExploreLifeLightFunctionSet",
        ) \
        .if_("dup_seed_photos == nil or #dup_seed_photos == 0") \
          .return_() \
        .end_() \
        .get_remote_embedding_lite(
          kess_service = "{{mmu_sim_emb_dup_i2i_embedding_service_name}}",
          shard_num = 4,
          id_converter = {"type_name": "kuibaEmbeddingIdConverter"},
          input_attr_name = "dup_seed_photos",
          output_attr_name = "dup_mmu_embeddings",
          query_source_type = "common_attr",
          size = 64,
          client_side_shard = True
        ) \
        .if_("dup_mmu_embeddings == nil or #dup_mmu_embeddings ~= 64 * #dup_seed_photos") \
          .return_() \
        .end_() \
        .set_attr_value(
          common_attrs = [
            {
              "name": "dup_mmu_embedding_dim",
              "type": "int",
              "value": 64
            }
          ]
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "dup_seed_photos", "as": "trigger_list"},
            {"name": "dup_mmu_embeddings", "as": "trigger_embedding_list"},
            {"name": "dup_mmu_embedding_dim", "as": "dim"}
          ],
          export_common_attr = [
            {"name": "trigger_list", "as": "dup_seed_photos"},
            {"name": "trigger_embedding_list", "as": "dup_mmu_embeddings"}
          ],
          function_name = "GetValidEmbeddings",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .retrieve_by_ann_embedding(
          kess_service = "{{mmu_sim_emb_dup_i2i_ann_service_name}}",
          space = "cosine",
          timeout_ms = 50,
          reason = self.reason,
          shard_num = 1,
          items_from_attr = ["dup_seed_photos"],
          embeddings_from_attr = ["dup_mmu_embeddings"],
          bound_type = {
            "total_limit": "{{retrieve_num}}",
          },
          algo_type = {
            "scann": {},
          },
          src_bucket = "photo",
          dest_bucket = "{{mmu_sim_emb_dup_i2i_retr_dest_bucket}}",
          save_distance_to_attr = "ann_dist_list"
        ) \
        .deduplicate() \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "ann_dist_threshold"
          ],
          import_item_attr = [
            "ann_dist_list"
          ],
          export_common_attr = [
            {"name": "remain_pid_list", "as": "dup_pid_list"}
          ],
          function_name = "GetPidListWithAnnDistThreshold",
          class_name = "ExploreLifeLightFunctionSet"
        ) \
        .limit(size = 0) \
      .else_() \
        .copy_attr(
          attrs = [
            {
              "from_common": "browse_screen__pid_list",
              "to_common": "browse_screen__pid_list_reversed"
            },
            {
              "from_common": "hate_list",
              "to_common": "hate_list_reversed"
            }
          ]
        ) \
        .explore_life_reverse_list_attr(
          common_attrs = [
            "browse_screen__pid_list_reversed",
            "hate_list_reversed"
          ]
        ) \
        .pack_common_attr(
          input_common_attrs = [
            "browse_screen__pid_list_reversed",
            "hate_list_reversed"
          ],
          output_common_attr = "dup_seed_photos",
          deduplicate = True,
          limit_num = "{{mmu_sim_emb_dup_i2i_seed_photo_size}}"
        ) \
        .if_("dup_seed_photos == nil or #dup_seed_photos == 0") \
          .return_() \
        .end_() \
        .get_remote_embedding_lite(
          kess_service = "{{mmu_sim_emb_dup_i2i_embedding_service_name}}",
          shard_num = 4,
          id_converter = {"type_name": "kuibaEmbeddingIdConverter"},
          input_attr_name = "dup_seed_photos",
          output_attr_name = "dup_mmu_embeddings",
          query_source_type = "common_attr",
          size = 64,
          client_side_shard = True
        ) \
        .if_("dup_mmu_embeddings == nil or #dup_mmu_embeddings ~= 64 * #dup_seed_photos") \
          .return_() \
        .end_() \
        .set_attr_value(
          common_attrs = [
            {
              "name": "dup_mmu_embedding_dim",
              "type": "int",
              "value": 64
            }
          ]
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "dup_seed_photos", "as": "trigger_list"},
            {"name": "dup_mmu_embeddings", "as": "trigger_embedding_list"},
            {"name": "dup_mmu_embedding_dim", "as": "dim"}
          ],
          export_common_attr = [
            {"name": "trigger_list", "as": "dup_seed_photos"},
            {"name": "trigger_embedding_list", "as": "dup_mmu_embeddings"}
          ],
          function_name = "GetValidEmbeddings",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .if_("dup_mmu_embeddings == nil or #dup_mmu_embeddings ~= 64 * #dup_seed_photos") \
          .return_() \
        .end_() \
        .retrieve_by_ann_embedding(
          kess_service = "{{mmu_sim_emb_dup_i2i_ann_service_name}}",
          space = "cosine",
          timeout_ms = 100,
          reason = self.reason,
          shard_num = 1,
          items_from_attr = ["dup_seed_photos"],
          embeddings_from_attr = ["dup_mmu_embeddings"],
          bound_type = {
            "total_limit": "{{mmu_sim_emb_dup_i2i_result_num}}",
          },
          algo_type = {
            "scann": {},
          },
          src_bucket = "photo",
          dest_bucket = "{{mmu_sim_emb_dup_i2i_retr_dest_bucket}}",
          save_distance_to_attr = "ann_dist_list"
        ) \
        .deduplicate() \
        .copy_item_meta_info(
          save_item_key_to_attr="photo_id",
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "ann_dist_threshold"
          ],
          import_item_attr = [
            "photo_id",
            "ann_dist_list"
          ],
          export_common_attr = [
            {"name": "remain_pid_list", "as": "dup_pid_list"}
          ],
          export_item_attr = [
            "ann_dist"
          ],
          function_name = "CalcAnnScoreWithThreshold",
          class_name = "ExploreLifeLightFunctionSet"
        ) \
        .limit(size = 0) \
      .end_()

  def post_process(self) -> None:
    self.flow \
      .log_debug_info(
        common_attrs = ["dup_pid_list"],
        for_debug_request_only = True
      )