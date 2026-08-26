from retrieval.retrieval_module import RetrievalModule

class SplashTagnexT2iRetrievalModule(RetrievalModule):
  def __init__(self, name=str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .if_("source_hetu_tag_level_info_hetu_tag == nil or #source_hetu_tag_level_info_hetu_tag == 0") \
        .return_() \
      .end_() \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "source_hetu_tag_level_info_hetu_tag"
        ],
        export_common_attr = [
          {"name": "tagnex_lv3_query_list", "as": "tagnex_lv3_query"},
          {"name": "tagnex_lv4_query_list", "as": "tagnex_lv4_query"},
          {"name": "tagnex_lv3_5_query_list", "as": "tagnex_lv3_5_query"}
        ],
        function_name = "BuildTagnexQueryList",
        class_name = "ExploreLightFunctionSetV2"
      ) \
      .if_("enable_tagnex_lv3_trigger ~= nil and enable_tagnex_lv3_trigger > 0 and "
           "tagnex_lv3_query ~= nil and #tagnex_lv3_query > 0") \
        .retrieve_by_remote_colossusdb_index(
          client_kconf = "colossus.inverted_index_kconf_client.explore_lbs_retr_index_client",
          reason = self.reason,
          querys = [
            {
              "query": "texNexLevel3:{{tagnex_lv3_query}}",
              "search_num": "{{search_num}}"
            }
          ],
        ) \
      .end_() \
      .if_("enable_tagnex_lv4_trigger ~= nil and enable_tagnex_lv4_trigger > 0 and "
           "tagnex_lv4_query ~= nil and #tagnex_lv4_query > 0") \
        .retrieve_by_remote_colossusdb_index(
          client_kconf = "colossus.inverted_index_kconf_client.explore_lbs_retr_index_client",
          reason = self.reason,
          querys = [
            {
              "query": "texNexLevel4:{{tagnex_lv4_query}}",
              "search_num": "{{search_num}}"
            }
          ],
        ) \
      .end_() \
      .if_("enable_tagnex_lv3_5_trigger ~= nil and enable_tagnex_lv3_5_trigger > 0 and "
           "tagnex_lv3_5_query ~= nil and #tagnex_lv3_5_query > 0") \
        .retrieve_by_remote_colossusdb_index(
          client_kconf = "colossus.inverted_index_kconf_client.explore_lbs_retr_index_client",
          reason = self.reason,
          querys = [
            {
              "query": "texNexLevel5:{{tagnex_lv3_5_query}}",
              "search_num": "{{search_num}}"
            }
          ],
        ) \
      .end_() \
      .deduplicate()\
      .filter_by_common_attr(
        common_attr = ["browse_screen__pid_list"]
      ) \
      .limit(
        size = "{{request_num}}"
      )
