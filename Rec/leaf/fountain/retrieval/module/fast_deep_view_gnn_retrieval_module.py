from retrieval.retrieval_module import RetrievalModule

class FastDeepViewGnnRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .retrieve_by_ann_embedding(
        kess_service = "{{fountain_fast_deepwalk_view_gnn_service}}",
        timeout_ms = 100,
        reason = self.reason,
        space = "ip",
        items_from_attr = ["featureFountainProfileEffViewPidList"],
        bound_type = {
          "top_k": "{{fountain_fast_deep_view_gnn_topk_num}}",
        },
        algo_type = {
          "scann": {},
        },
        src_bucket = "target_tensor",
        dest_bucket = "context_tensor",
        dest_bucket_item_type = 1,
        skip = "{{skip_fast_deep_view_gnn_retr}}") 