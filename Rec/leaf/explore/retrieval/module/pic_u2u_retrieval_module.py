from retrieval.retrieval_module import RetrievalModule

class PicU2uRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .if_("use_delegate == 1") \
        .explore_custom_trim_user_info(
          user_info_attr="userInfo",
          save_trimed_user_info_to_attr="trimed_user_info",
          trim_user_info=self.trim_user_info_attr_list
        ) \
        .delegate_retrieve(
          kess_service="{{delegate_service_name}}",
          timeout_ms=100,
          reason=self.reason,
          request_num="{{request_num}}",
          send_common_attrs_in_request=False,
          send_common_attrs=[
            {"name": "trimed_user_info", "as": "user"},
          ] + self.send_common_attrs,
          reset_item_type=0,
        ) \
      .else_() \
        .if_("enable_u2u_ann_v2 == 1") \
          .retrieve_by_ann_embedding(
            reason=1,
            kess_service="{{ann_service_name_v2}}",
            space="ip",
            timeout_ms=25,
            items_from_attr=["_USER_ID_"],
            bound_type={
              "top_k": "{{ann_retr_top_k_v2}}"
            },
            algo_type={
              "scann": {}
            },
            src_data_type="{{ann_src_data_type_v2}}",
            src_bucket="{{ann_src_bucket_v2}}",
            dest_bucket="{{ann_dest_bucket_v2}}",
            save_result_to_common_attr="u2u_result_list"
          ) \
        .end_() \
        .if_("u2u_result_list == nil or #u2u_result_list <= 0") \
          .retrieve_by_ann_embedding(
            reason=1,
            kess_service="{{ann_service_name}}",
            space="cosine",
            timeout_ms=25,
            items_from_attr=["_USER_ID_"],
            bound_type={
              "top_k": "{{ann_retr_top_k}}"
            },
            algo_type={
              "scann": {}
            },
            src_data_type="{{ann_src_data_type}}",
            src_bucket="{{ann_src_bucket}}",
            dest_bucket="{{ann_dest_bucket}}",
            save_result_to_common_attr="u2u_result_list"
          ) \
        .end_() \
        .if_("shuffle_u2u_result == 1") \
          .shuffle_list_attr(
            common_attr="u2u_result_list"
          ) \
          .truncate(
            item_list_from_attr="u2u_result_list",
            size_limit="{{u2u_num_limit}}"
          ) \
        .end_() \
        .if_("u2i_from_index == 1") \
          .retrieve_by_remote_index(
            kess_service="{{u2i_index_service_name}}",
            timeout_ms=50,
            reason=self.reason,
            querys=[
                {
                    "query": "{{u2i_index_query_term}}:{{u2u_result_list}}",
                    "search_num": "{{u2i_index_search_num}}",
                }
            ]
          ) \
        .else_() \
          .retrieve_by_redis(
            reason=self.reason,
            cluster_name="recoAnalysis",
            timeout_ms=50,
            retrieve_num="{{u2i_retrieve_num}}",
            key_from_attr="u2u_result_list",
            key_prefix="{{u2i_redis_key_prefix}}",
            retrieve_num_per_key="{{u2i_retrieve_num_per_key}}",
            item_separator=","
          ) \
        .end_() \
      .end_() \
      .deduplicate() \
      .filter_by_browse_set() \
      .filter_by_common_attr(
        common_attr=["browse_screen__pid_list"]
      ) \
      .shuffle() \
      .limit(size="{{request_num}}")

  @property
  def trim_user_info_attr_list(self) -> list:
    return self.config.get("trim_user_info_attr_list", [])

  @property
  def send_common_attrs(self) -> list:
    return self.config.get("send_common_attrs", [])
