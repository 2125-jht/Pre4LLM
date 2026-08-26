from cascading import CommonModule

class CascadingExploreSimilarUsersHetuTagsEnricher(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self):
    self.flow \
    .if_("enable_explore_similar_users_hetu_tags_enricher == 1") \
      .pack_common_attr(
        input_common_attrs = ["sim_user_list"],
        output_common_attr = "explore_sim_user_list",
        limit_num = "{{explore_similar_users_num_limit}}",
        deduplicate = True,
      ) \
      .colossus(
        service_name = "grpc_colossusSimV2",
        client_type = "common_item_client",
        output_attr = "similar_user_colossus_list",
        input_attr = "explore_sim_user_list",
        max_resp_item_num = 50,
        is_batch_query = True,
        parse_to_pb = False,
      ) \
      .explore_explore_similar_users_hetu_enricher(
        enable_user_retrieval_result_explore = "{{explore_similar_users_explore_use_retrieval_result}}",
        save_cluster_id_to_attr = "similar_user_explore_hetu_tags",
        explore_interest_cnt = "{{explore_similar_users_explore_cluster_count_limit}}",
        explore_interest_limit = "{{explore_similar_users_explore_play_count_limit}}",
        similar_user_colossus_attr = "similar_user_colossus_list",
        similar_user_list_attr = "explore_sim_user_list",
        colossuse_min_play_time = "{{explore_similar_users_explore_min_play_time}}"
      ) \
    .end_() \
    .pack_common_attr(
      input_common_attrs = ["colossus_explore_hetu_tags", "similar_user_explore_hetu_tags"],
      output_common_attr = "explore_hetu_tags",
      deduplicate = True,
    )

  def post_process(self) -> None:
    self.flow \
      .log_debug_info(
        common_attrs = [
          "explore_hetu_tags",
        ],
      )
