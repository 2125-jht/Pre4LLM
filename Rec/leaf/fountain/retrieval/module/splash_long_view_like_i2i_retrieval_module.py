from retrieval.retrieval_module import RetrievalModule

class LongViewLikeI2IRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .retrieve_by_ann_embedding(
        kess_service = "{{fountain_splash_long_view_like_i2i_retr_kess_service}}",
        space = "ip",
        timeout_ms = 50,
        reason = self.reason,
        shard_num = 1,
        items_from_attr = ["featureSourcePId"],
        bound_type = {
          "total_limit": "{{fountain_splash_long_view_like_i2i_retr_num}}",
        },
        algo_type = {
          "scann": {},
        },
        src_bucket = "{{fountain_splash_long_view_like_i2i_retr_src_bucket}}",
        dest_bucket = "{{fountain_splash_long_view_like_i2i_retr_dest_bucket}}",
        dest_bucket_item_type = 0,
        skip = "{{skip_fountain_splash_long_view_like_i2i_retr}}"
      )