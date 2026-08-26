from retrieval.retrieval_module import RetrievalModule

class SplashEmbI2iRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .retrieve_by_ann_embedding(
        kess_service = "{{fountain_emb_i2i_retr_kess_name_splash}}",
        timeout_ms = 150,
        reason = self.reason,
        space = "ip",
        items_from_attr = ["commonRetrievalPhotos"],
        attr_single_limit = 100,
        bound_type = {
          "total_limit": "{{fountain_reco_emb_i2i_retr_num_splash}}",
        },
        algo_type = {
          "faiss": {}
        },
        src_bucket = "mio_item",
        dest_bucket = "fountain_common_pid_bucket",
        dest_bucket_item_type = 1,
        skip = "{{fountain_skip_reco_emb_i2i_retr_splash}}")
