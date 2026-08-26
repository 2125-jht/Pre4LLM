from retrieval import RetrievalModule
class ExploreMllmU2iRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
    
  def process(self) -> None:
    self.flow \
      .if_("_USER_ID_ <= 0") \
        .return_() \
      .end_() \
      .get_remote_embedding_lite_v2(
        kess_service = "grpc-MLLM-uemb-kess",
        id_converter = {"type_name": "kuibaEmbeddingIdConverter"},
        shard_num = 2,
        size = 128,
        query_source_type = "common_attr",
        input_attr_name = "_USER_ID_",
        output_attr_name = "trigger_embedding",
        client_side_shard = True,
        slot = 667,
        is_raw_data = True,
        raw_data_type = 'float32',
        timeout_ms = 20,
      )\
      .if_("trigger_embedding == nil") \
        .return_() \
      .end_() \
      .retrieve_by_ann_embedding(
        reason = self.reason,
        kess_service = "{{service_name}}",
        timeout_ms = 70,
        shard_num = 1,
        space = 'ip',
        items_from_attr = ["_USER_ID_"],
        embeddings_from_attr = ['trigger_embedding'],
        bound_type = {
          "total_limit": "{{result_num}}",
        },
        algo_type = {
          "scann": {},
        },
        # item 对应的数据桶，如 i2i 这里是 "item", u2i 这里是 "user"
        src_bucket = "{{src_bucket}}",
        # 目标桶
        dest_bucket = "{{dest_bucket}}",
        ) \
      .deduplicate() \
      .shuffle() \
      .limit(1000)