from retrieval.retrieval_module import RetrievalModule

class AcfRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .enrich_with_protobuf(
        from_extra_var = "user_info_ptr",
        attrs = [
          dict(name="followList", path="follow_list.user.id"),
          dict(name="clickAidList", path="user_profile_v1.click_list.author_id"),
          dict(name="likeAidList", path="user_profile_v1.like_list.author_id"),
          dict(name="forwardAidList", path="user_profile_v1.forward_list.author_id"),
          dict(name="profileAidList", path="user_profile_v1.profile_enter_list.author_id"),
          dict(name="downloadAidList", path="user_profile_v1.download_video_list.author_id"),
          dict(name="collectAidList", path="user_profile_v1.collect_list.author_id"),
          dict(name="hateAidList", path="user_profile_v1.hate_list.author_id"),
          dict(name="videoPlayRawAids", path="user_profile_v1.video_playing_stat.author_id"),
          dict(name="videoDurations", path="user_profile_v1.video_playing_stat.video_duration"),
          dict(name="videoPlayTime", path="user_profile_v1.video_playing_stat.playing_time")
        ]
      ) \
      .if_("enable_rand_source > 0") \
        .shuffle_list_attr(common_attr = "followList") \
        .shuffle_list_attr(common_attr = "clickAidList") \
        .shuffle_list_attr(common_attr = "likeAidList") \
        .shuffle_list_attr(common_attr = "forwardAidList") \
        .shuffle_list_attr(common_attr = "profileAidList") \
        .shuffle_list_attr(common_attr = "downloadAidList") \
        .shuffle_list_attr(common_attr = "collectAidList") \
      .end_() \
      .enrich_attr_by_lua(
        import_common_attr = [
          "followList", "clickAidList", "likeAidList", "forwardAidList", 
          "profileAidList", "downloadAidList", "collectAidList", "hateAidList", "videoPlayRawAids", "videoDurations", "videoPlayTime", 
          "follow_source_num", "recent_source_num", "click_num", "like_num", "forward_num", "profile_num", "download_num", "collect_num"
        ],
        export_common_attr = ["source_aids"],
        function_for_common = "generate_triggers",
        lua_script_file = "explore/retrieval/lua/module/acf_retr__generate_trigger.lua"
      ) \
      .retrieve_by_redis(
        reason = 1,
        retrieve_num = 10000,
        retrieve_num_per_key = "{{sim_num_each}}",
        cluster_name = "recoUserPreference",
        timeout_ms = 20,
        key_from_attr = "source_aids",
        key_prefix = "{{sim_redis_prefix}}",
        item_separator = ",",
        attr_separator = ":",
        extra_item_attrs = [
          {"name": "sim_score", "type": "double"}
        ],
        save_result_to_common_attr = "sim_authors"
      ) \
      .filter_by_attr(
        item_list_from_attr = "sim_authors",
        attr_name = "sim_score",
        remove_if = "<",
        compare_to = "{{min_sim_score}}",
        remove_if_attr_missing = True
      ) \
      .filter_by_common_attr(
        item_list_from_attr = "sim_authors",
        common_attr = ["source_aids", "hateAidList"]
      ) \
      .if_("sim_authors == nil or #sim_authors == 0") \
        .return_() \
      .end_() \
      .if_("enable_rand_sim > 0") \
        .shuffle_list_attr(common_attr = "sim_authors") \
      .end_() \
      .limit(
        item_list_from_attr = "sim_authors",
        size = "{{sim_author_limit}}"
      ) \
      .pack_item_attr(
        item_source = {"common_attr": ["sim_authors"]},
        mappings = [
          {"aggregator": "concat", "from_item_attr": "sim_score", "to_common_attr": "sim_score_list", "default_val": -1.0}
        ]
      ) \
      .retrieve_by_remote_index(
        kess_service = "{{index_service_name}}",
        timeout_ms = "{{index_timeout_ms}}",
        reason = 1,
        querys = [
          {
            "query": self.remote_index_query_term + ":{{sim_authors}}",
            "random_search": 0,
            "search_num": "{{search_num_per_author}}"
          }
        ],
        save_score_to_attr = "index_score",
        save_query_index_to_attr = "index_src",
        save_result_to_common_attr = "result_item_id"
      ) \
      .deduplicate(
        item_list_from_attr = "result_item_id",
      ) \
      .explore_add_inverted_index_weighted_score(
        item_list_from_attr = "result_item_id",
        score_attr = "index_score",
        query_index_attr = "index_src",
        query_index_weight_list = "{{sim_score_list}}",
        save_weighted_score_to_attr = "weighted_score",
      ) \
      .sort(
        item_list_from_attr = "result_item_id",
        score_from_attr = "weighted_score",
      ) \
      .retrieve_by_common_attrs(
        attrs = [
          {
            "name": "result_item_id",
            "reason": self.reason,
            "num_limit": 5000
          }
        ]
      ) \
      .limit("{{retrieve_num}}")
  
  @property
  def remote_index_query_term(self) -> str:
    return self.config.get("remote_index_query_term", "authorId2PhotoIdOrderByUploadTime")
