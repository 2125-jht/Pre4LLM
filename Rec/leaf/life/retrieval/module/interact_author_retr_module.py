from retrieval.retrieval_module import RetrievalModule

class InteractAuthorRetrModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self):
    self.flow \
      .if_("enable_retrieval == 0") \
        .return_() \
      .end_() \
      .enrich_with_protobuf(
        from_extra_var = "user_info_ptr",
        attrs = [
          dict(name="downloadAids", path="user_profile_v1.download_video_list.author_id", repeat_limit={"user_profile_v1.download_video_list": 10}),
          dict(name="searchClickAids", path="user_profile_v1.search_click_author_list.author_id", repeat_limit={"user_profile_v1.search_click_author_list": 100}),
          dict(name="dupClickAids", path="user_profile_v1.dup_click_list.author_id", repeat_limit={"user_profile_v1.dup_click_list": 3}),
          dict(name="videoPlayRawAids", path="user_profile_v1.video_playing_stat.author_id"),
          dict(name="videoDurations", path="user_profile_v1.video_playing_stat.video_duration"),
          dict(name="videoPlayTime", path="user_profile_v1.video_playing_stat.playing_time"), 
          dict(name="profileEnterAids", path="user_profile_v1.profile_enter_list.author_id", repeat_limit={"user_profile_v1.profile_enter_list": 200}),
          dict(name="likeAids", path="user_profile_v1.like_list.author_id", repeat_limit={"user_profile_v1.like_list": 150}),
          dict(name="forwardAids", path="user_profile_v1.forward_list.author_id", repeat_limit={"user_profile_v1.forward_list": 100}),
          dict(name="commentAids", path="user_profile_v1.comment_list.author_id", repeat_limit={"user_profile_v1.comment_list": 200}),
          dict(name="hateAids", path="user_profile_v1.hate_list.author_id")
        ]
      ) \
      .enrich_attr_by_lua(
        import_common_attr = ["videoPlayRawAids", "videoDurations", "videoPlayTime", "duration_lower_limit", "duration_upper_limit"], 
        export_common_attr = ["longViewAids"], 
        function_for_common = "calculate",
        lua_script_file = "life/retrieval/lua/module/interact_author_retr__fetch_longview_trigger.lua"
      ) \
      .pack_common_attr(
        input_common_attrs = ["hateAids", "browse_screen__aid_list"],
        output_common_attr = "hateAids",
        deduplicate = True
      ) \
      .enrich_attr_by_lua(
        import_common_attr = ["downloadAids", "searchClickAids", "dupClickAids", "longViewAids", "profileEnterAids", "likeAids", "forwardAids", "commentAids", "hateAids"],
        export_common_attr = ["triggerAids"],
        function_for_common = "calculate",
        lua_script_file = "life/retrieval/lua/module/interact_author_retr__trigger_filter.lua"
      ) \
      .if_("enable_trigger_shuffle == 1") \
        .shuffle_list_attr(common_attr = "triggerAids") \
      .end_() \
      .retrieve_by_remote_index(
        kess_service = "{{remote_index_service_name}}",
        timeout_ms = "{{remote_index_service_timeout_ms}}",
        reason = self.reason, 
        querys = [
          {
            "query": "{{remote_index_query_term}}:{{triggerAids}}",
            "search_num": "{{remote_index_search_num}}", 
            "max_attr_num": 1000
          }
        ],
        save_score_to_attr = "index_score"
      ) \
      .deduplicate() \
      .filter_by_common_attr(
        common_attr = ["browse_screen__pid_list"],
        skip = "{{skip_filter_by_browse_screen}}"
      ) \
      .filter_by_browse_set(
        skip = "{{skip_browse_set}}"
      ) \
      .sort(
        score_from_attr = "index_score"
      ) \
      .limit(
        size = "{{result_num}}"
      )
  
  def post_process(self) -> None:
    self.flow \
      .log_debug_info(
        common_attrs = ["downloadAids", "searchClickAids", "dupClickAids", "longViewAids", "profileEnterAids", "likeAids", "forwardAids", "commentAids", "hateAids", "triggerAids"]
      )