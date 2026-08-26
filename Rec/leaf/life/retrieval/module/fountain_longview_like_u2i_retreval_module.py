from retrieval.retrieval_module import RetrievalModule

class FountainLongViewLikeU2IRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self):
    self.flow \
      .switch_("request_type") \
        .case_("fountain_fast_v1_life") \
          .retrieve_by_ann_embedding(
            kess_service = "{{xlife_fountain_long_view_like_retrieval_kess_service}}",
            space = "cosine",
            timeout_ms = 100,
            reason = self.reason,
            items_from_attr = ["_USER_ID_"],
            attr_single_limit = 50,
            bound_type = {
              "total_limit": "{{xlife_fountain_long_view_like_retrieval_total_limit}}",
            },
            algo_type = {
              "scann": {}
            },
            src_bucket = "user",
            dest_bucket = "photo",
            dest_bucket_item_type = 1
          ) \
          .deduplicate() \
          .filter_by_common_attr(
            common_attr=["browse_screen__pid_list"]
          ) \
      .end_()