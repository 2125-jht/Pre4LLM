from retrieval.retrieval_module import RetrievalModule

class SplashGnnI2iRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .retrieve_by_ann_embedding(
        kess_service = "grpc_fountainGnnI2IAnnServer",
        timeout_ms = 150,
        reason = self.reason,
        space = "ip",
        items_from_attr = ["commonRetrievalPhotos"],
        bound_type = {
          "total_limit": "{{fountain_gnn_i2i_retrieval_splash_num}}",
        },
        algo_type = {
          "faiss": {},
        },
        src_bucket = "target_tensor",
        dest_bucket = "target_tensor",
        dest_bucket_item_type = 1,
        skip = "{{skip_fountain_gnn_i2i_retrieval_splash}}")

      