from retrieval import CommonModule

class ExploreExploreHetuTagsEnricherModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self):
    self.flow \
    .if_("enable_explore_hetu_tags_enricher == 1") \
      .explore_explore_hetu_tags_enricher(
        colossus_v2_attr_name = "colossus_resp_v2",
        user_info_ptr_attr = "user_info_ptr",
        shortterm_hetu_attr = "interest_explore_shortterm_hetu",
        longterm_hetu_one_attr = "interest_explore_longterm_hetu_one",
        longterm_hetu_two_attr = "interest_explore_longterm_hetu_two",
        export_explore_hetu_tags_attr = "colossus_explore_hetu_tags",
        recent_top_show_hetu_attr = "recent_top_show_hetu",
        explore_cluster_count_limit = "{{explore_explore_cluster_count_limit}}",
        enable_explore_cluster_target = "{{explore_enable_explore_cluster_target}}",
        enable_recent_stat_only_explore = "{{explore_enable_recent_stat_only_explore}}",
        explore_cluster_target_count_limit = "{{explore_explore_cluster_target_count_limit}}",
        high_quality_tags_attr = "high_quality_tags",
        explore_cluster_recent_play_count_limit = "{{explore_explore_cluster_recent_play_count_limit}}",
        explore_cluster_recent_show_count_limit = "{{explore_explore_cluster_recent_show_count_limit}}",
        explore_cluster_play_count_limit = "{{explore_explore_cluster_play_count_limit}}",
        explore_cluster_stat_time_hour_limit = "{{explore_explore_cluster_stat_time_hour_limit}}",
        explore_cluster_recent_play_top_ratio = "{{explore_explore_cluster_recent_play_top_ratio}}",
        explore_cluster_recent_show_top_ratio = "{{explore_explore_cluster_recent_show_top_ratio}}",
        enable_explore_user_explore_coeff = "{{explore_enable_explore_user_explore_coeff}}",
        explore_colossus_min_size = "{{explore_explore_colossus_min_size}}",
        explore_cluster_interest_score_limit = "{{explore_explore_cluster_interest_score_limit}}",
        explore_cluster_already_interest_score_limit = "{{explore_explore_cluster_already_interest_score_limit}}",
        explore_user_explore_avg_value = "{{explore_explore_user_explore_avg_value}}",
        explore_user_explore_coeff_min = "{{explore_explore_user_explore_coeff_min}}",
        explore_user_explore_coeff_max = "{{explore_explore_user_explore_coeff_max}}",
        interest_score_click_pow_weight = "{{explore_interest_score_click_pow_weight}}",
        interest_score_lv_weight = "{{explore_interest_score_lv_weight}}",
        interest_score_like_weight = "{{explore_interest_score_like_weight}}",
        interest_score_enter_profile_weight = "{{explore_interest_score_enter_profile_weight}}",
        interest_score_comment_weight = "{{explore_interest_score_comment_weight}}",
        interest_score_follow_weight = "{{explore_interest_score_follow_weight}}",
        interest_score_forward_weight = "{{explore_interest_score_forward_weight}}",
        interest_score_playtime_weight = "{{explore_interest_score_playtime_weight}}",
        enable_explore_cluster_ignore_interest_hetu = "{{explore_enable_explore_cluster_ignore_interest_hetu}}"
      ) \
    .end_()

  def post_process(self) -> None:
    self.flow \
      .log_debug_info(
        common_attrs = [
          "colossus_explore_hetu_tags",
          "recent_top_show_hetu",
        ],
      )
