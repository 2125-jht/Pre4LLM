from retrieval.retrieval_module import RetrievalModule

class MmuU2uRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .retrieve_by_ann_embedding(
        reason=1,
        kess_service="{{ann_service_name}}",
        space="cosine",
        timeout_ms=50,
        items_from_attr=["_USER_ID_"],
        bound_type={
          "top_k": "{{retr_top_k}}"
        },
        algo_type={
          "scann": {}
        },
        src_data_type="{{ann_src_data_type}}",
        src_bucket="{{ann_src_data_type}}",
        dest_bucket="{{ann_dest_bucket}}",
        save_result_to_common_attr="u2u_result_list"
      ) \
      .retrieve_by_redis(
        reason=self.reason,
        cluster_name="recoAnalysis",
        timeout_ms=20,
        retrieve_num="{{user_photo_retrieve_num}}",
        key_from_attr="u2u_result_list",
        key_prefix="{{user_photo_redis_key_prefix}}",
        retrieve_num_per_key="{{user_photo_retrieve_num_per_key}}",
        item_separator=","
      ) \
      .deduplicate() \
      .filter_by_common_attr(
        common_attr=["browse_screen__pid_list"]
      ) \
      .limit(
        size="{{retrieve_num}}"
      )
