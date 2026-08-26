from retrieval.retrieval_module import RetrievalModule

class AnnU2iRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
    .if_("enable_emb_server ~= nil and enable_emb_server == 0") \
      .retrieve_by_ann_embedding(
        reason = self.reason,
        kess_service = "{{ann_service}}",
        space = self.space,
        timeout_ms = 100,
        items_from_attr = ["_USER_ID_"],
        bound_type = {
          "total_limit": "{{retrieve_num}}"
        },
        algo_type = {
          "scann": {}
        },
        src_data_type = "user",
        src_bucket = "user",
        dest_bucket = "photo"
      ) \
    .else_() \
      .get_remote_embedding(
        kess_service = "{{embedding_service}}",
        timeout_ms = 10,
        id_converter = {
          "type_name": "mioEmbeddingIdConverter"
        },
        slot = self.slot,
        save_to_common_attr = True,
        output_embedding_list_attr = "user_embedding_list",
        query_source_type = "user_id",
        is_raw_data = False,
        is_raw_data_list = False
      ) \
      .retrieve_by_ann_embedding(
        reason = self.reason,
        kess_service = "{{ann_service}}",
        space = self.space,
        timeout_ms = 100,
        items_from_attr = ["_USER_ID_"],
        embeddings_from_attr = ["user_embedding_list"],
        bound_type = {
          "total_limit": "{{retrieve_num}}"
        },
        algo_type = {
          "scann": {}
        },
        src_data_type = "user",
        src_bucket = "user",
        dest_bucket = "photo"
      ) \
    .end_() \
    .filter_by_common_attr(
      common_attr = ["browse_screen__pid_list"]
    )

  @property
  def slot(self) -> int:
    assert "slot" in self.config
    return self.config["slot"]

  @property
  def space(self) -> str:
    assert "space" in self.config
    return self.config["space"]