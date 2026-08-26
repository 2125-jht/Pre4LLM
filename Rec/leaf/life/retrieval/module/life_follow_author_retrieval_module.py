from retrieval import RetrievalModule

class LifeFollowAuthorRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  @classmethod
  def is_retrieval(cls) -> bool:
    return True
  
  def process(self) -> None:
    self.flow \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "like_aids",
          "follow_aids",
          "forward_aids",
          "comment_aids",
          "collect_aids",
          "like_timestamps",
          "forward_timestamps",
          "comment_timestamps",
          "collect_timestamps",
          "playstat_playtimes",
          "playstat_durations",
          "profile_v1_click_trigger_aids",
          "userRecentViewTimeListRaw",
          "life_follow_author_time_gap"
        ],
        export_common_attr = [
          "followAidsTrigger"
        ],
        function_name = "ParseUserFollowAuthorList",
        class_name = "ExploreLifeLightFunctionSet"
      ) \
      .if_("enable_life_follow_author_insert_retr_expand == 1 and uLifePreferAuthor ~= nil and (life_follow_author_insert_retr_expand_limit_low_active ~= 1 or uIsLifeHighActive ~= 1)") \
        .pack_common_attr(
          input_common_attrs = ["followAidsTrigger", "uLifePreferAuthor"],
          output_common_attr = "followAidsTrigger",
          deduplicate = True,
          limit_num = "{{life_follow_author_insert_retr_trigger_limit}}",
        ) \
      .end_() \
      .if_("enable_life_follow_author_insert_retr_expand_colossus == 1 and (life_follow_author_insert_retr_expand_colossus_limit_low_active ~= 1 or uIsLifeHighActive ~= 1)") \
        .explore_life_colossus_author_enricher(
          colossus_resp_attr = "colossus_resp_v2",
          time_window_day = "{{life_colossus_prefer_author_time_win_day}}",
          author_score_thresh = "{{life_colossus_prefer_author_score_thresh}}",
          save_author_count = "{{life_colossus_prefer_author_count}}",
          save_interest_authors_attr = "colossus_prefer_authors"
        ) \
        .pack_common_attr(
          input_common_attrs = ["followAidsTrigger", "colossus_prefer_authors"],
          output_common_attr = "followAidsTrigger",
          deduplicate = True,
          limit_num = "{{life_follow_author_insert_retr_trigger_limit}}",
        ) \
      .end_() \
      .if_("enable_life_follow_author_insert_retr_expand_recent == 1 and (life_follow_author_insert_retr_expand_recent_limit_low_active ~= 1 or uIsLifeHighActive ~= 1)") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "like_aids",
            "follow_aids",
            "forward_aids",
            "comment_aids",
            "collect_aids",
            "like_timestamps",
            "follow_timestamps",
            "forward_timestamps",
            "comment_timestamps",
            "collect_timestamps",
            "playstat_playtimes",
            "playstat_durations",
            {"name": "profile_v1_click_trigger_aids", "as": "playstat_aids"},
            {"name": "userRecentViewTimeListRaw", "as": "playstat_timestamps"},
            {"name": "life_recent_author_gap_time_minute", "as": "gap_time_minute"},
            {"name": "life_recent_author_max_num", "as": "recent_author_max_num"}
          ],
          export_common_attr = [
            "recent_author_list"
          ],
          function_name = "ParseUserRecentAuthorList",
          class_name = "ExploreLifeLightFunctionSet"
        ) \
        .pack_common_attr(
          input_common_attrs = ["followAidsTrigger", "recent_author_list"],
          output_common_attr = "followAidsTrigger",
          deduplicate = True,
        ) \
      .end_() \
      .retrieve_by_redis(
        reason = self.reason,
        cluster_name="recoAnalysis",
        retrieve_num = "{{life_follow_author_insert_retrieval_num}}",
        retrieve_num_per_key = "{{life_follow_author_insert_retrieval_per_key}}",
        key_from_attr = "followAidsTrigger",
        timeout_ms = 50,
        item_separator="," 
      ) \
      .filter_by_common_attr(
        common_attr = ["browse_screen__pid_list"]
      )