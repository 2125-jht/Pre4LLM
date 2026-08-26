from retrieval.retrieval_module import RetrievalModule

class McBucketU2iRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  @classmethod
  def is_retrieval(cls) -> bool:
    return True
    
  def process(self) -> None:
    ann_args = {
      "kess_service": "{{ann_service}}",
      "space": "ip",
      "timeout_ms": 20,
      "reason": self.reason,
      "shard_num": 1,
      "items_from_attr": ["_USER_ID_"],
      "embeddings_from_attr": ["mc_u2i_user_embedding_list"],
      "algo_type": {
        "scann": {},
      },
      "src_bucket": "{{ann_src_bucket}}",
      "dest_bucket_item_type": 0
    }
    # TODO(wangjia07): use retrieval leaf
    self.flow \
      .if_ ("user_risk_level and user_risk_level < risk_level_min or mc_u2i_user_embedding_list == nil") \
        .return_() \
      .end_() \
      .retrieve_by_ann_embedding(
        **ann_args,
        dest_bucket = "{{ann_dest_bucket0}}",
        bound_type = {"total_limit": "{{retr_num_bucket0}}"}
      ) \
      .retrieve_by_ann_embedding(
        **ann_args,
        dest_bucket = "{{ann_dest_bucket1}}",
        bound_type = {"total_limit": "{{retr_num_bucket1}}"}
      ) \
      .retrieve_by_ann_embedding(
        **ann_args,
        dest_bucket = "{{ann_dest_bucket2}}",
        bound_type = {"total_limit": "{{retr_num_bucket2}}"}
      ) \
      .retrieve_by_ann_embedding(
        **ann_args,
        dest_bucket = "{{ann_dest_bucket3}}",
        bound_type = {"total_limit": "{{retr_num_bucket3}}"}
      ) \
      .deduplicate()