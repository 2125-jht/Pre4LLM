from retrieval.retrieval_module import RetrievalModule

class PicVid2picRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .pack_common_attr(
        input_common_attrs = ["collect_list","follow_list","like_list","click_list","forward_list"],
        output_common_attr = "trigger_list",
        deduplicate = True,
        limit_num = "{{trigger_limit}}",
      ) \
      .explore_retrieve_by_redis_list_range(
        reason = self.reason,
        key_attr = "trigger_list",
        save_score_to_attr = "index_score",
        cluster_name = "{{redis_cluster_name}}",
        timeout_ms = 100,
        key_prefix = "{{redis_key_prefix}}",
        retrieve_num_per_key = "{{retrieve_num_per_key}}",
        retrieve_num = "{{redis_retrieve_num}}",
        cal_score_type =1,
      ) \
      .filter_by_common_attr(
        common_attr = ["browse_screen__pid_list"]
      ) \
      .deduplicate() \
      .sort(
        score_from_attr = "index_score",
      ) \
      .limit(size = "{{retr_total_limit}}")