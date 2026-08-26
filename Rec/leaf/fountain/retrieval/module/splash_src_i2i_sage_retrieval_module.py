from retrieval.retrieval_module import RetrievalModule

class SrcI2iSageRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .retrieve_by_ann_embedding(
        kess_service = "{{fountain_splash_src_i2i_sage}}",
        timeout_ms = 150,
        reason = self.reason,
        space = "ip",
        items_from_attr = ["commonRetrievalPhotos"],
        bound_type = {
          "total_limit": "{{splash_src_i2i_sage_num}}",
        },
        algo_type = {
          "scann": {}
        },
        src_data_type = "input_node",
        src_bucket = "input_node",
        dest_bucket = "pos_node",
        dest_bucket_item_type = 1,
        skip = "{{skip_splash_src_i2i_sage_retrieval}}")
    