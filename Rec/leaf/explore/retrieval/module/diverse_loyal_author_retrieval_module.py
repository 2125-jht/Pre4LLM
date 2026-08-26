from retrieval.retrieval_module import RetrievalModule

class DiverseLoyalAuthorRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self):
    self.flow \
      .copy_attr(
        attrs = [{
          "from_common": "uFansList",
          "to_common": "loyalfans_aids"
        },
        {
          "from_common": "a2i_interact_aids",
          "to_common": "interact_aids"
        },
        {
          "from_common": "commonTriggerAids",
          "to_common": "commontrigger_aids"
        },
        {
          "from_common": "explore_la_long_view_author_list",
          "to_common": "longview_aids"
        },
        {
          "from_common": "friendAids",
          "to_common": "friend_aids"
        },
        {
          "from_common": "followAids",
          "to_common": "ori_follow_aids"
        },
        {
          "from_common": "follow_aids",
          "to_common": "hot_follow_aids"
        },
        ]
      ) \
      .shuffle_list_attr(
        common_attr = "loyalfans_aids"
      ) \
      .truncate(
        size_limit = "{{loyalfans_aids_num}}",
        item_list_from_attr = "loyalfans_aids"
      ) \
      .shuffle_list_attr(
        common_attr = "interact_aids"
      ) \
      .truncate(
        size_limit = "{{interact_aids_num}}",
        item_list_from_attr = "interact_aids"
      ) \
      .shuffle_list_attr(
        common_attr = "commontrigger_aids"
      ) \
      .truncate(
        size_limit = "{{common_trigger_num}}",
        item_list_from_attr = "commontrigger_aids"
      ) \
      .shuffle_list_attr(
        common_attr = "longview_aids"
      ) \
      .truncate(
        size_limit = "{{longview_aids_num}}",
        item_list_from_attr = "longview_aids"
      ) \
      .shuffle_list_attr(
        common_attr = "friend_aids"
      ) \
      .truncate(
        size_limit = "{{friend_aids_num}}",
        item_list_from_attr = "friend_aids"
      ) \
      .shuffle_list_attr(
        common_attr = "ori_follow_aids"
      ) \
      .truncate(
        size_limit = "{{follow_aids_num}}",
        item_list_from_attr = "ori_follow_aids"
      ) \
      .shuffle_list_attr(
        common_attr = "hot_follow_aids"
      ) \
      .truncate(
        size_limit = "{{hot_follow_aids_num}}",
        item_list_from_attr = "hot_follow_aids"
      ) \
      .retrieve_by_redis(
        reason = 0,
        retrieve_num = "{{profile_aid_retr_by_redis_retrieval_num}}",
        cluster_name = "recoUserPreference",
        timeout_ms = 10,
        key_from_attr = "_USER_ID_", 
        key_prefix = "User_Prefer_Author_",
        item_separator = ",",
        save_result_to_common_attr = "longterm_worth_profile_aids"
      ) \
      .shuffle_list_attr(
        common_attr = "longterm_worth_profile_aids"
      ) \
      .truncate(
        size_limit = "{{longterm_worth_profile_aids_num}}",
        item_list_from_attr = "longterm_worth_profile_aids"
      ) \
      .retrieve_by_redis(
        reason = 0,
        retrieve_num = "{{hot_retr_by_redis_retrieval_num}}",
        cluster_name = "dataScienceExp1",
        timeout_ms = 10,
        key_from_attr = "_USER_ID_", 
        key_prefix = "User_Hpage_Prefer_",
        item_separator = ",",
        save_result_to_common_attr = "longterm_worth_hot_aids"
      ) \
      .shuffle_list_attr(
        common_attr = "longterm_worth_hot_aids"
      ) \
      .truncate(
        size_limit = "{{longterm_worth_hot_aids_num}}",
        item_list_from_attr = "longterm_worth_hot_aids"
      ) \
      .pack_common_attr(
        input_common_attrs = ["loyalfans_aids",
          "interact_aids",
          "commontrigger_aids",
          "longview_aids",
          "friend_aids",
          "ori_follow_aids",
          "hot_follow_aids",
          "longterm_worth_profile_aids",
          "longterm_worth_hot_aids"
        ],
        output_common_attr = "diverseLoyalTriggerAids",
        deduplicate = True
      ) \
      .if_("enable_retr_by_ann_embedding == 1") \
        .retrieve_by_ann_embedding(
          kess_service = "{{ann_service}}",
          timeout_ms = 40,
          reason = 1,
          items_from_attr = ["diverseLoyalTriggerAids"],
          bound_type = {
            "top_k": "{{sim_author_num}}",
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
      .else_() \
        .copy_attr(
          attrs = [{
            "from_common": "diverseLoyalTriggerAids",
            "to_common": "sim_aids"
          }]
        ) \
      .end_() \
      .deduplicate(
        item_list_from_attr = "sim_aids",
      ) \
      .if_("enable_trigger_shuffle == 1") \
        .shuffle_list_attr(
          common_attr = "sim_aids"
        ) \
      .end_() \
      .retrieve_by_remote_index(
        kess_service = "{{remote_index_service_name}}",
        timeout_ms = 40,
        reason = self.reason, 
        querys = [
          {
            "query": "{{retr_index_term}}:{{sim_aids}}",
            "search_num": "{{remote_index_search_num}}", 
            "max_attr_num": "{{loyal_author_max_num}}",
            "random_search" : "{{enable_random_search}}"
          }
        ],
        default_search_num = 200,
        default_total_request_num = 1000,
        default_random_search = 1,
      ) \
      .if_("enable_diverse_loyal_browset_filter == 1") \
        .filter_by_browse_set() \
      .end_() \
      .deduplicate() \
      .limit(
        size = "{{result_num}}"
      )