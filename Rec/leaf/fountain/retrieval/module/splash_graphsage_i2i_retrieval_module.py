from retrieval.retrieval_module import RetrievalModule

class GraphsageI2iRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .retrieve_by_ann_embedding(
        kess_service="{{fountain_graphsage_i2i_retr_kess_name_splash}}",
        timeout_ms=100,
        reason=self.reason,
        space="cosine",
        items_from_attr=["commonRetrievalPhotos"],
        bound_type={
          "total_limit": "{{fountain_graphsage_i2i_retr_num_splash}}",
        },
        algo_type={
          "scann": {},
        },
        src_bucket="photo",
        dest_bucket="photo_bucket_splash",
        dest_bucket_item_type=1,
        skip="{{fountain_skip_graphsage_i2i_retr_splash}}")