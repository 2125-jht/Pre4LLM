from retrieval.retrieval_module import RetrievalModule

class MCU2ILifeContentRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self):
    self.flow \
      .if_("skip_fountain_mc_u2i_retrieval == 0 or skip_fountain_mc_u2i_life_content_retrieval == 0", to_be_delete = "date=2024-05-29;committer=huzengyi") \
        .get_remote_embedding(
          kess_service = "{{fountain_mc_user_emb_service_name}}",
          shard_num = 1,
          timeout_ms = "{{fountain_mc_u2i_timeout}}",
          id_converter = {
            "type_name": "kuibaEmbeddingIdConverter"
          },
          save_to_common_attr = True,
          output_item_list_attr = "user_id_list",
          output_embedding_list_attr = "user_embedding_list",
          query_source_type = "user_id",
          raw_data_type = "uint16",
          client_side_shard = False,
          is_raw_data=False,
          is_raw_data_list=False,
        ) \
      .end_() \
      .retrieve_by_ann_embedding(
        kess_service="{{fountain_mc_u2i_life_content_service}}",
        timeout_ms = 100,
        reason = self.reason,
        space = "ip",
        items_from_attr = ["user_id_list"],
        embeddings_from_attr = ["user_embedding_list"],
        bound_type = {
          "total_limit": "{{fountain_mc_u2i_life_content_retrieval_num}}",
        },
        algo_type = {
          "scann": {},
        },
        src_bucket = "photo",
        dest_bucket = "{{fountain_mc_u2i_life_content_dest_photo_bucket}}",
        dest_bucket_item_type = 0,
        skip = "{{skip_fountain_mc_u2i_life_content_retrieval}}",
      )