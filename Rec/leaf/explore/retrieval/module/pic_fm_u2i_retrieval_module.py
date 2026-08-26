from retrieval.retrieval_module import RetrievalModule

class PicFmU2IRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .retrieve_by_ann_embedding(
        reason = self.reason,
        kess_service = "{{ann_service_name}}",
        space = "{{ann_retr_space}}",
        timeout_ms = "{{ann_timeout_ms}}",
        items_from_attr = ["_USER_ID_"],
        bound_type = {
          "top_k": "{{ann_user_top_k}}"
        },
        algo_type = {
          "scann": {}
        },
        src_data_type = "{{ann_src_data_type}}",
        src_bucket = "{{ann_src_data_type}}",
        dest_bucket = "{{ann_dest_bucket}}",
      ) \
      .deduplicate() \
      .filter_by_common_attr(
        common_attr = ["browse_screen__pid_list"]
      )
