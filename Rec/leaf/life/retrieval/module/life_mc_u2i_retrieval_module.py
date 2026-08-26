from retrieval.retrieval_module import RetrievalModule

class LifeMcU2iRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  @classmethod
  def is_retrieval(cls) -> bool:
    return True
    
  def process(self) -> None:
    self.flow \
      .retrieve_by_ann_embedding(
        kess_service = "{{life_mc_u2i_ann_service_name}}",
        space = "ip",
        timeout_ms = 100,
        reason = self.reason,
        shard_num = 1,
        items_from_attr = ["_USER_ID_"],
        embeddings_from_attr = ["mc_u2i_user_embedding_list"],
        bound_type = {
        "total_limit": "{{life_mc_u2i_ann_service_retr_count}}",
        },
        algo_type = {
        "scann": {},
        },
        src_bucket = "{{life_mc_u2i_ann_service_bucket}}",
        dest_bucket = "{{life_mc_u2i_ann_service_bucket}}",
        dest_bucket_item_type = 0
      ) \
      .deduplicate()

  def post_process(self) -> None:
    pass