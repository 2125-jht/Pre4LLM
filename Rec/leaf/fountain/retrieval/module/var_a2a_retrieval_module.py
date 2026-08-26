from retrieval.retrieval_module import RetrievalModule

class VarA2aRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .copy_attr(
        attrs = [{
          "from_common": "video_playing_stat_aid_list",
          "to_common": "realtime_aids"
        }]
      ) \
      .pack_common_attr(
        input_common_attrs = ["featureFountainProfileLikeAidList",  "featureFountainProfileFollowAidList", "featureFountainProfileLongViewAidList"],
        output_common_attr = "interact_aids",
        deduplicate = True
      ) \
      .shuffle_list_attr(
        common_attr = "interact_aids"
      ) \
      .shuffle_list_attr(
        common_attr = "realtime_aids"
      ) \
      .truncate(
        size_limit = "{{fountain_var_a2a_interact_num}}",
        item_list_from_attr = "interact_aids"
      ) \
      .truncate(
        size_limit = "{{fountain_var_a2a_realtime_num}}",
        item_list_from_attr = "realtime_aids"
      ) \
      .retrieve_by_redis(
        reason = 0,
        retrieve_num = "{{fountain_var_a2a_longterm_author_num}}",
        cluster_name = "recoEyeshotFollow",
        timeout_ms = 20,
        key_from_attr = "_USER_ID_", 
        key_prefix = "efirst_",
        item_separator = ",",
        save_result_to_common_attr = "long_term_author"
      ) \
      .pack_common_attr(
        input_common_attrs = ["realtime_aids", "interact_aids", "long_term_author"],
        output_common_attr = "trigger_aids",
        deduplicate = True
      ) \
      .retrieve_by_ann_embedding(
        kess_service = "{{fountain_var_a2a_retr_ann_service}}",
        timeout_ms = 40,
        reason = 1,
        items_from_attr = ["trigger_aids"],
        bound_type = {
          "top_k": "{{fountain_var_a2a_sim_author_num}}",
        },
        algo_type = {
          "scann": {}
        },
        space = "cosine",
        src_data_type = "author",
        src_bucket = "author",
        dest_bucket = "author_bucket",
        save_result_to_common_attr = "sim_aids"
      ) \
      .deduplicate(
        item_list_from_attr = "sim_aids",
      ) \
      .retrieve_by_remote_index(
        kess_service = "{{fountain_var_a2a_retr_index_service}}",
        timeout_ms = 40,
        reason = self.reason,
        querys = [
          {
            "query": "{{fountain_var_a2a_ordering}}:{{sim_aids}}",
            "search_num": "{{fountain_var_a2a_num_per_author}}",
            "random_search": 0
          }
        ]
      ) \
      .deduplicate() \
      .limit("{{fountain_var_a2a_cand_num}}")
