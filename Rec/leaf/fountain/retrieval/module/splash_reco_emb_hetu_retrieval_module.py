from retrieval.retrieval_module import RetrievalModule

class RecoEmbHetuRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .log_debug_info(
          common_attrs = [
              "fountain_reco_emb_hetu_retr_kess_name_splash",
              "fountain_reco_emb_hetu_retrieval_splash_num",
              "topSubdivisionHetuBucket",
              "skip_fountain_reco_emb_hetu_retrieval_splash"
          ],
          for_debug_request_only = True
      ) \
      .retrieve_by_ann_embedding(
        kess_service = "{{fountain_reco_emb_hetu_retr_kess_name_splash}}",
        timeout_ms = 150,
        reason = self.reason,
        space = "cosine",
        items_from_attr = ["colossusRetrievalTrigger"],
        bound_type = {
          "top_k": "{{fountain_reco_emb_hetu_retrieval_splash_num}}",
        },
        algo_type = {
          "faiss": {}
        },
        src_bucket = "mio_item",
        dest_bucket = "{{topSubdivisionHetuBucket}}",
        dest_bucket_item_type = 1,
        skip = "{{skip_fountain_reco_emb_hetu_retrieval_splash}}")
