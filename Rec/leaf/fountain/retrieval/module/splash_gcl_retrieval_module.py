from retrieval.retrieval_module import RetrievalModule

class SplashGclRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .retrieve_by_ann_embedding(
        kess_service = "{{fountain_gcl_retr_splash_ann_service}}",
        timeout_ms = 50,
        reason = self.reason,
        items_from_attr = ["commonRetrievalPhotos"],
        attr_single_limit = 100,
        bound_type = {
          "total_limit": "{{fountain_gcl_retr_splash_request_num}}",
        },
        algo_type = {
          "scann": {},
        },
        src_bucket = "photo",
        dest_bucket = "photo_bucket_splash",
        skip = "{{skip_fountain_gcl_retr_splash}}")