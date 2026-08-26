from retrieval.retrieval_module import RetrievalModule

class Cb2cfI2iRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .retrieve_by_ann_embedding(
        kess_service="{{fountain_cb2cf_i2i_retr_splash_service_name}}",
        timeout_ms=100,
        reason=self.reason,
        space="ip",
        items_from_attr=["commonRetrievalPhotos"],
        bound_type={
          "total_limit": "{{fountain_cb2cf_i2i_retr_splash_retr_num}}",
        },
        algo_type = {
          "faiss": {}
        },
        attr_single_limit=50,
        src_data_type="photo",
        src_bucket="photo",
        dest_bucket="photo_bucket",
        dest_bucket_item_type=1,
        skip="{{skip_fountain_cb2cf_i2i_retr_splash}}"
      )