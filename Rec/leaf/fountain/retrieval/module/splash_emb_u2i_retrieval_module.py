from retrieval.retrieval_module import RetrievalModule

class SplashEmbU2iRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .retrieve_by_ann_embedding(
        kess_service = "grpc_fountainEmbU2IServer",
        timeout_ms = 150,
        reason = 10143,
        space = "ip",
        items_from_attr = ["_USER_ID_"],
        attr_single_limit = 100,
        bound_type = {
          "total_limit": "{{fountain_reco_emb_u2i_retr_num_splash}}",
        },
        algo_type = {
          "faiss": {}
        },
        src_bucket = "mio_user",
        dest_bucket = "fountain_common_pid_bucket",
        dest_bucket_item_type = 1,
        skip = "{{fountain_skip_reco_emb_u2i_retr_splash}}")