from retrieval.retrieval_module import RetrievalModule

class CrossoverU2u2iRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .retrieve_by_ann_embedding(
        reason = self.reason,
        kess_service="grpc_mmuUserEmbU2UServer",
        user_key_for_embedding="uid_only",
        space="ip",
        timeout_ms=10,
        items_from_attr=["_USER_ID_"],
        bound_type={
          "top_k": 100
        },
        algo_type={
          "scann": {}
        },
        src_data_type="user",
        src_bucket="user",
        dest_bucket="index_user",
        save_result_to_common_attr="life_crossover_user_list"
      ) \
      .if_("enable_life_crossover_explore == 1") \
        .retrieve_by_ann_embedding(
          reason = self.reason,
          kess_service="grpc_exploreMmuU2uServer",
          space="cosine",
          timeout_ms=10,
          items_from_attr=["_USER_ID_"],
          bound_type={
            "top_k": 100
          },
          algo_type={
            "scann": {}
          },
          src_data_type="user",
          src_bucket="user",
          dest_bucket="explore_pic_users",
          save_result_to_common_attr="life_crossover_explore_user_list"
        ) \
        .pack_common_attr(
            input_common_attrs = ["life_crossover_user_list", "life_crossover_explore_user_list"],
            output_common_attr = "life_crossover_user_list",
            deduplicate = True
          ) \
      .end_() \
      .retrieve_by_redis(
          reason =self.reason,
          retrieve_num = 500,
          # retrieve_num_per_key = 50,
          cluster_name = "recoExploreNegPhoto",
          timeout_ms = 10,
          key_from_attr = "life_crossover_user_list",
          key_prefix = "life_rerank_pos_",
          item_separator = ",",
          # save_result_to_common_attr = "life_crossover_item_list"
        )  \
      .if_("enable_life_crossover_colossus == 1") \
        .explore_retrieve_by_redis_list_range(
            reason =self.reason,
            retrieve_num = 500,
            # retrieve_num_per_key = 50,
            cluster_name = "recoColossusTriggers",
            timeout_ms = 10,
            key_attr = "life_crossover_user_list",
            key_prefix = "clicks_",
            item_separator = ",",
            # save_result_to_common_attr = "life_crossover_colossus_item_list"
          ) \
          .pack_common_attr(
            input_common_attrs = ["life_crossover_item_list", "life_crossover_colossus_item_list"],
            output_common_attr = "life_crossover_item_list",
            deduplicate = True
          ) \
      .end_() \
      .if_("enable_trigger_shuffle == 1") \
        .shuffle_list_attr(
          common_attr = "life_crossover_item_list"
        ) \
      .end_() \
      .deduplicate() \
      .limit("{{retrieve_num}}")

  def post_process(self):
    self.flow \
      .log_debug_info(
        common_attrs = [ 
                        "duration_ths", "max_interact_aids_num", 
                        "max_interest_auhtor_hpage__author_num", "max_interest_auhtor_ppage__author_num",
                        "max_long_view_aid_num", "max_source_aids_num",
                        "playtime_ths", "redis_prefix", "retr_num_each", "retr_num_limit",
                        "life_crossover_user_list",
                        "life_crossover_item_list",
                        "life_crossover_u2u2i_retr__result"],
          for_debug_request_only = True,
      )