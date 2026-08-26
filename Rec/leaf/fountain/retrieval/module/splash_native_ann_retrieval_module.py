from retrieval.retrieval_module import RetrievalModule

class NativeAnnRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .retrieve_by_ann_embedding(
        kess_service = "{{fountain_splash_native_ann_service}}",
        timeout_ms = 50,
        reason = self.reason,
        items_from_attr = ["featureSourcePId"],
        bound_type = {
          "total_limit": "{{fountain_splash_native_retr_num}}",
        },
        algo_type = {
          "scann": {},
        },
        src_data_type = "{{fountain_splash_native_retr_src_type}}",
        src_bucket = "{{fountain_splash_native_retr_src_type}}",
        dest_bucket = "{{fountain_splash_native_retr_dest_bucket}}",
        dest_bucket_item_type = 1,
        skip = "{{skip_fountain_splash_native_retr}}",
      )
