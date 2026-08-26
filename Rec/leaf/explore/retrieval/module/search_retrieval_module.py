from retrieval.retrieval_module import RetrievalModule

class SearchRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self):
    self.flow \
      .explore_retr_search_trigger_enriche(
        user_info_ptr_attr = "user_info_ptr",
        search_emb_explore_retr_max_time_decay = "{{search_emb_explore_retr_max_time_decay}}",
        ht_negative_feedback_timeout_min_in_search_retr = "{{ht_negative_feedback_timeout_min_in_search_retr}}",
        enable_search_retr_walk_off_by_not_click = "{{enable_search_retr_walk_off_by_not_click}}",
        search_negative_feedback_filter_threshold = "{{search_negative_feedback_filter_threshold}}",
        enable_search_retr_only_select_user = "{{enable_search_retr_only_select_user}}",
        search_retr_high_full_active_or_mid_search = "{{search_retr_high_full_active_or_mid_search}}",
        search_retr_high_full_active_and_mid_search = "{{search_retr_high_full_active_and_mid_search}}",
        user_active_degree_down = "{{user_active_degree_down}}",
        user_active_degree_up = "{{user_active_degree_up}}",
        search_day_count_down = "{{search_day_count_down}}",
        search_day_count_up = "{{search_day_count_up}}",
        enable_search_back_explore_cal = "{{enable_search_back_explore_cal}}",
        search_to_current_interval = "{{search_to_current_interval}}",
        backexplore_to_current_interval = "{{backexplore_to_current_interval}}",
        output_search_retr_on_attr = "output_search_retr_on",
        output_search_author_retr_on_attr = "output_search_author_retr_on",
        output_search_aid_trigger_attr = "output_search_aid_trigger"
      ) \
      .if_("output_search_retr_on ~= nil and output_search_retr_on == 1") \
        .explore_custom_trim_user_info(
          user_info_attr = "userInfo",
          save_trimed_user_info_to_attr = "trimedUserInfo",
          trim_user_info = [
            "user_profile_v1.real_show_list",
            "user_profile_v1.click_list",
            "user_profile_v1.hate_list",
            "user_profile_v1.video_playing_stat",
            "user_profile_v1.search_photo_play_list",
            "user_profile_v1.search_query_list",
          ]
        ) \
        .if_("trimedUserInfo ~= nil") \
          .delegate_retrieve(# 搜索quary,item召回
            kess_service = "{{kess_service}}",
            request_type = "default",
            request_num = 1200,
            timeout_ms = 100,
            reason = self.reason,
            send_browse_set = True,
            send_common_attrs_in_request = False,
            send_common_attrs = [
              {"name": "trimedUserInfo", "as": "user"},
            ]) \
        .end_if_() \
      .end_if_() \
      .if_("output_search_author_retr_on ~= nil and output_search_author_retr_on == 1 and output_search_aid_trigger ~= nil") \
        .retrieve_by_remote_index(# 搜索作者召回
          kess_service = "{{author_kess_service}}",
          timeout_ms = 150,
          reason = self.reason,
          common_query = "",
          querys = [{
            "query": "aId:{{output_search_aid_trigger}}",
            "search_num": 50,
          }],
          attr_single_limit = 200,
          default_search_num = 50,
          default_random_search = 1,
          default_total_request_num = 50,) \
      .end_if_() \
      .deduplicate()

  @property
  def total_limit(self):
    return self.config.get("total_limit", 1200)

  @property
  def retrieval_tag(self):
    return self.config.get("retrieval_tag", "explore_search_retr")