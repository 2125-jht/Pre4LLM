from retrieval.retrieval_module import RetrievalModule

class FreshRecoEmbu2iRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .retrieve_by_ann_embedding(
        kess_service = "{{fountain_fresh_reco_embu2i_retrieval_kess_service}}",
        space = "cosine",
        timeout_ms = 100,
        reason = self.reason,
        items_from_attr = ["_USER_ID_"],
        attr_single_limit = 50,
        bound_type = {
          "total_limit": "{{fountain_fresh_reco_embu2i_retrieval_total_limit}}",
        },
        algo_type = {
          "scann": {}
        },
        src_bucket = "user",
        dest_bucket = "{{fountain_fresh_reco_embu2i_retrieval_bucket}}",
        dest_bucket_item_type = 1,
        skip = "{{fountain_fresh_reco_embu2i_retrieval_skip}}"
      )
