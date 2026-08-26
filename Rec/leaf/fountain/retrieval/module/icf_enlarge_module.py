from retrieval import CommonModule

# TODO 暂时拆出来
class IcfEnlargeModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self):
    self.flow \
      .enrich_attr_by_lua(
        import_common_attr = [
          "featureFountainProfileEffViewPidList",
          "featureUserHateList",
          "skip_fountain_eff_view_filter",
          "colossusRetrievalTrigger",
          "skip_fountain_colossus_filter",
          "featureFountainProfileLongViewPidList",
          "skip_fountain_longview_filter"],
        export_common_attr = [
          "featureFountainProfileEffViewPidList",
          "colossusRetrievalTrigger",
          "featureFountainProfileLongViewPidListSub"
        ],
        function_for_common = "filter_trigger_list",
        lua_script_file = "fountain/retrieval/lua/module/icf_enlarge__calc_icf_enlarge_trigger.lua",
      ) \
      .copy_attr(
        attrs=[{
          "from_common": "featureUserProfileV1ProfileEnterPidList",
          "to_common": "featureUserProfileV1ProfileEnterPidListLite"
        }],
        skip="{{skip_fountain_profile_enter_limit}}") \
      .limit(
        item_list_from_attr="featureUserProfileV1ProfileEnterPidListLite",
        size="{{fountain_profile_enter_limit}}",
        skip="{{skip_fountain_profile_enter_limit}}") \
      .enrich_attr_by_lua(
        import_common_attr = [
          "featureUserProfileV1LikePidList",
          "featureUserProfileV1FollowPidList",
          "featureUserProfileV1CommentPidList",
          "featureUserProfileV1ForwardPidList",
          "featureUserProfileV1ProfileEnterPidListLite",
          "add_fountain_profile_for_interaction_list",
          "fountainActionTriggers"],
        export_common_attr = [
          "shuffle_interaction_photo_list"
        ],
        function_for_common = "shuffle_interaction_list",
        lua_script_file = "fountain/retrieval/lua/module/icf_enlarge__calc_icf_enlarge_trigger.lua",
        skip = "{{skip_interact_shuffle}}") \
      .deduplicate(
        item_list_from_attr="shuffle_interaction_photo_list",
        skip="{{skip_interaction_list_dedup}}")