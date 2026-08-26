from retrieval.retrieval_module import RetrievalModule

class TopSubdivisionNnRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .retrieve_by_ann_embedding(
        kess_service = "grpc_fountainNNRetrHetuTagServer",
        timeout_ms = 150,
        reason = self.reason,
        space = "cosine",
        items_from_attr = ["featureFountainProfileEffViewPidList"],
        bound_type = {
          "top_k": "{{fountain_top_subdivision_nn_retrieval_tag_splash_num}}",
        },
        algo_type = {
          "faiss": {}
        },
        src_bucket = "mio_item",
        dest_bucket = "{{topSubdivisionHetuBucket}}",
        dest_bucket_item_type = 1,
        skip = "{{skip_fountain_top_subdivision_nn_retrieval_tag_splash}}")