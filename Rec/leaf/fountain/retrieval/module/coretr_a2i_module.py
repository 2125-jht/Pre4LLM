from retrieval.retrieval_module import RetrievalModule

class CoretrA2iRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .if_("skip_fountain_coretr_a2i_retr == 0") \
        .enrich_attr_by_lua(
          import_common_attr = [
            "featureFountainProfileLikeAidList",
            "featureFountainProfileFollowAidList",
            "featureFountainProfileLongViewAidList",
            "featureFountainProfileShortViewAidList",
            "featureUserProfileV1FollowAidList",
            "featureUserProfileV1ProfileEnterAidList",
            "hateAidList"
          ],
          export_common_attr = ["sim_author_retr_src_authors"],
          function_for_common = "fetch_src_authors",
          lua_script_file = "fountain/retrieval/lua/module/coretr_a2i_retr__fetch_src_authors.lua") \
        .shuffle_list_attr(
          common_attr = "sim_author_retr_src_authors") \
        .pack_common_attr(
          input_common_attrs = [
          "featureUserProfileV1LikeAidList",
          "featureUserProfileV1CommentAidList",
          "featureUserProfileV1DownloadAidList",
          "featureUserProfileV1FollowAidList",
          "featureUserProfileV1ForwardAidList",
          "featureUserProfileV1ProfileEnterAidList",
          "user_fountain_forward_aid_list",
          "user_fountain_follow_aid_list",
          "user_fountain_like_aid_list",
          "user_fountain_comment_aid_list"
          ],
          output_common_attr = "interact_aids",
          deduplicate = True
        ) \
        .shuffle_list_attr(common_attr="interact_aids") \
        .pack_common_attr(
          input_common_attrs = [
            "interact_aids"
          ],
          output_common_attr = "interact_aids",
          limit_num = "{{fountain_coretr_a2i_retr_max_interact_aids_num}}"
        ) \
        .pack_common_attr(
          input_common_attrs = [
            "sim_author_retr_src_authors"
          ],
          output_common_attr = "sim_src_aids",
          limit_num = "{{fountain_coretr_a2i_retr_max_sim_src_aids_num}}",
          deduplicate = True
        ) \
        .pack_common_attr(
          input_common_attrs = [
            "sim_src_aids",
            "interact_aids"
          ],
          output_common_attr = "source_aids",
          deduplicate = True
        ) \
        .retrieve_by_redis(
          cluster_name = "recoUserPreference",
          reason = self.reason,
          retrieve_num = "{{fountain_coretr_a2i_retr_max_num}}",
          retrieve_num_per_key = "{{fountain_coretr_a2i_retr_num_per_key}}",
          timeout_ms = 30,
          key_from_attr = "source_aids",
          key_prefix = "{{fountain_coretr_a2i_retr_key_prefix}}",
          item_separator = ",",
          attr_separator = ":",
          extra_item_attrs = [
            {"name": "redis_score"}
          ]
        ) \
      .end_()