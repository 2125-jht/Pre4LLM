from retrieval.retrieval_module import RetrievalModule

class FastNodeViewGnnRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .retrieve_by_ann_embedding(
        kess_service = "{{fountain_fast_node_view_gnn}}",
        timeout_ms = 100,
        reason = self.reason,
        space = "ip",
        items_from_attr = ["featureFountainProfileEffViewPidList"],
        bound_type = {
          "top_k": "{{fast_node_view_topk_num}}",
        },
        algo_type = {
          "scann": {}
        },
        src_data_type = "input_node",
        src_bucket = "input_node",
        dest_bucket = "pos_node",
        dest_bucket_item_type = 1,
        skip = "{{skip_fast_node_view_gnn_retrieval}}")