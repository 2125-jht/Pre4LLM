from retrieval.retrieval_module import RetrievalModule

class McSeEmbeddingRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .retrieve_by_ann_embedding(
        skip = "{{skip_fountain_mc_se_embedding_retrieval_splash}}",
        kess_service = "{{fountain_mc_se_embedding_retrieval_service_splash}}",
        timeout_ms = 150,
        reason = self.reason,
        space = "cosine",
        attr_single_limit = 30,
        items_from_attr = ["commonRetrievalPhotos"],
        bound_type = {
          "total_limit": "{{fountain_mc_se_embedding_retrieval_num_splash}}",
        },
        algo_type = {
          "scann": {},
        },
        src_bucket = "photo",
        dest_bucket = "{{fountain_mc_se_embedding_retrieval_dest_bucket}}",
        dest_bucket_item_type = 0,
      )