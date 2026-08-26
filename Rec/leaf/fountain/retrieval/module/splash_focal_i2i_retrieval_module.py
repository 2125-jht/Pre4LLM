from retrieval.retrieval_module import RetrievalModule

class FocalI2iRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .retrieve_by_ann_embedding(
        kess_service = "{{fountain_splash_focal_retr_ann_service}}",
        timeout_ms = 150,
        reason = self.reason,
        items_from_attr = ["commonRetrievalPhotos"],
        attr_single_limit = 100,
        bound_type = {
          "total_limit": "{{fountain_splash_focal_retr_cand_num}}",
        },
        algo_type = {
          "scann": {},
        },
        src_bucket = "photo",
        dest_bucket = "photo",
        dest_bucket_item_type = 1,
        skip = "{{skip_fountain_splash_focal_retr}}",
      ) 