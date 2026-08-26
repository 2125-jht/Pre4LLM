from retrieval import CommonModule

class McUserEmbeddingModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
    
  def process(self) -> None:
    self.flow \
      .get_remote_embedding_lite(
        kess_service = "{{embedding_service_name}}",
        shard_num = 1,
        timeout_ms = 10,
        id_converter = {
          "type_name": "kuibaEmbeddingIdConverter"
        },
        size = 128,
        output_attr_name = "mc_u2i_user_embedding_list",
        query_source_type = "user_id",
        client_side_shard = True
      )