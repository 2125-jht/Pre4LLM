from retrieval.retrieval_module import RetrievalModule

class PicVid2picRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
        .if_("enable_global_trigger ~= nil and enable_global_trigger > 0")
    ## 从 global trigger 里完成抽取
    self._sample_pic_global_triggers("trigger_list", "trigger_weight_list")
    self.flow \
      .else_() \
        .pack_common_attr(
          input_common_attrs = ["collect_list","follow_list","like_list","click_list","forward_list"],
          output_common_attr = "trigger_list",
          deduplicate = True,
          limit_num = "{{trigger_limit}}",
        ) \
      .end_() \
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