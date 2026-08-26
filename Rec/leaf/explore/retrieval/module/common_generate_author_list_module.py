from retrieval import CommonModule

class CommonGenarateAuthorListModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self):
    self.flow \
      .copy_attr(
        attrs = [
          {"from_common": "profile_v1_click_trigger_aids", "to_common": "videoPlayRawAids"},
          {"from_common": "playstat_durations", "to_common": "videoDurations"},
          {"from_common": "playstat_playtimes", "to_common": "videoPlayTime"},
          {"from_common": "hate_aids", "to_common": "hateAids"},
        ]
      ) \
      .enrich_attr_by_lua(
        import_common_attr = ["videoPlayRawAids", "videoDurations", "videoPlayTime", "duration_lower_limit", "duration_upper_limit"], 
        export_common_attr = ["longViewAids"], 
        function_for_common = "calculate",
        lua_script_file = "explore/retrieval/lua/module/interact_author_retr__fetch_longview_trigger.lua"
      ) \
      .pack_common_attr(
        input_common_attrs = ["hateAids", "browse_screen__aid_list"],
        output_common_attr = "hateAids",
        deduplicate = True
      ) \
      .enrich_attr_by_lua(
        import_common_attr = ["downloadAids", "searchClickAids", "dupClickAids", "longViewAids", "profileEnterAids", "likeAids", "forwardAids", "commentAids", "hateAids"],
        export_common_attr = ["commonTriggerAids"],
        function_for_common = "calculate",
        lua_script_file = "explore/retrieval/lua/module/interact_author_retr__trigger_filter.lua"
      ) \
      .pack_common_attr(
        input_common_attrs = [
          "like_aids",
          "follow_aids",
          "forward_aids",
          "comment_aids",
          "collect_aids"
        ],
        output_common_attr = "a2i_interact_aids",
        deduplicate = True
      )