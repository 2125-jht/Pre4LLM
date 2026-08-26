from retrieval.retrieval_module import RetrievalModule

class A2iRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .pack_common_attr(
        input_common_attrs = [
          "like_aids",
          "follow_aids",
          "forward_aids",
          "comment_aids",
          "collect_aids"
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
        limit_num = "{{max_interact_aids_num}}"
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "profile_v1_click_trigger_aids", "as": "aid_list"},
          {"name": "playstat_durations", "as": "duration_list"},
          {"name": "playstat_playtimes", "as": "playtime_list"},
          "duration_ths",
          "playtime_ths"
        ],
        export_common_attr = [
          {"name": "interset_aids", "as": "long_view_aids"},
        ],
        function_name = "SelectInterestAuthor",
        class_name = "ExploreLightFunctionSetV2"
      ) \
      .copy_attr(
        attrs = [
          {
            "from_common": "interest_auhtor_ppage__author_list",
            "to_common": "interest_auhtor_ppage"
          },
          {
            "from_common": "interest_auhtor_hpage__author_list",
            "to_common": "interest_auhtor_hpage"
          }]
      ) \
      .shuffle_list_attr(
        common_attr = "interest_auhtor_ppage"
      ) \
      .shuffle_list_attr(
        common_attr = "interest_auhtor_hpage"
      ) \
      .pack_common_attr(
        input_common_attrs = [
          "interest_auhtor_ppage"
        ],
        output_common_attr = "interest_auhtor_ppage",
        limit_num = "{{max_interest_auhtor_ppage__author_num}}"
      ) \
      .pack_common_attr(
        input_common_attrs = [
          "interest_auhtor_hpage"
        ],
        output_common_attr = "interest_auhtor_hpage",
        limit_num = "{{max_interest_auhtor_hpage__author_num}}"
      ) \
      .pack_common_attr(
        input_common_attrs = [
          "long_view_aids"
        ],
        output_common_attr = "long_view_aids",
        limit_num = "{{max_long_view_aid_num}}"
      ) \
      .pack_common_attr(
        input_common_attrs = [
          "interact_aids",
          "long_view_aids",
          "interest_auhtor_ppage",
          "interest_auhtor_hpage"
        ],
        output_common_attr = "source_aids",
        limit_num = "{{max_source_aids_num}}",
        deduplicate = True
      ) \
      .if_("enable_trigger_shuffle == 1") \
        .shuffle_list_attr(
          common_attr = "source_aids"
        ) \
      .end_() \
      .retrieve_by_redis(
        reason = self.reason,
        retrieve_num = "{{retr_num_limit}}",
        retrieve_num_per_key = "{{retr_num_each}}",
        cluster_name = "recoUserPreference",
        timeout_ms = 20,
        key_from_attr = "source_aids",
        key_prefix = "{{redis_prefix}}",
        item_separator = ",",
        attr_separator = ":",
        extra_item_attrs = [
          {"name": "sim_score", "type": "double"}
        ],
      ) \
      .deduplicate() \
      .sort(
        score_from_attr = "sim_score",
      ) \
      .limit("{{retrieve_num}}")
