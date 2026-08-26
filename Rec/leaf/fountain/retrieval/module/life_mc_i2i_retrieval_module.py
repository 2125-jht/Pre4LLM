from retrieval.retrieval_module import RetrievalModule

class LifeMcI2IRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .if_("enable_life_fountain_global_trigger_selected == 1", to_be_delete = "date=2024-05-29;committer=denghong") \
        .explore_trigger_selected_enricher(
          user_info_ptr_attr = "userInfoPb",
          colossus_resp_attr = "colossus_resp_v2",
          enable_life_hetu_restrict = "{{use_fountain_hetu_restrict}}",
          final_item_list_attr = "fountain_life_selected_trigger_list",
          life_restrict_hetu_level1 = "{{fountain_life_restrict_hetu_level1}}"
        ) \
      .end_() \
      .if_("enable_fountain_life_mc_i2i_retrievel == 1") \
        .pack_common_attr(
          input_common_attrs = [
            "featureFountainProfileEffViewPidList",
            "commonRetrievalPhotos",
            "featureFountainProfileLikePidList",
            "featureFountainProfileCommentPidList",
            "featureFountainProfileFollowPidList",
            "fountain_life_selected_trigger_list"
          ],
          output_common_attr = "seed_photos",
          deduplicate = True
        ) \
        .shuffle_list_attr(
          common_attr= "seed_photos"
        ) \
        .pack_common_attr(
          input_common_attrs = ["seed_photos"],
          output_common_attr = "seed_photos",
          limit_num = "{{fountain_life_mc_i2i_seed_photo_size}}"
        ) \
        .get_remote_embedding(
          kess_service = "{{fountain_life_mc_i2i_embedding_service_name}}",
          shard_num = 8,
          timeout_ms = 20,
          slot = 0,
          id_converter = {
            "type_name": "kuibaEmbeddingIdConverter"
          },
          item_list_from_attr = "seed_photos",
          save_to_common_attr = True,
          output_item_list_attr = "fountain_life_mc_trigger_list",
          output_embedding_list_attr = "fountain_life_mc_trigger_embedding",
          query_source_type = "item_key",
          client_side_shard = True,
          raw_data_type = "uint16",
          is_raw_data = False,
          is_raw_data_list = False,
        ) \
        .retrieve_by_ann_embedding(
          kess_service = "{{fountain_life_mc_i2i_ann_service_name}}",
          space = "cosine",
          timeout_ms = 100,
          reason = self.reason,
          shard_num = 1,
          items_from_attr = ["fountain_life_mc_trigger_list"],
          embeddings_from_attr = ["fountain_life_mc_trigger_embedding"],
          bound_type = {
            "total_limit": "{{fountain_life_mc_i2i_result_num}}",
          },
          algo_type = {
            "scann": {},
          },
          src_bucket = "{{fountain_life_mc_i2i_src_bucket}}",
          dest_bucket = "{{fountain_life_mc_i2i_dest_bucket}}",
        ) \
      .end_()