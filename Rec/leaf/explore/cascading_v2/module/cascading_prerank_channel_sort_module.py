from cascading_v2.module.channel.prerank_photo_channel import PrerankPhotoChannelParitioner, PrerankPhotoChannelScorer
from cascading_v2.module.channel.prerank_picture_channel import PrerankPictureChannelParitioner, PrerankPictureChannelScorer
from cascading_v2.module.channel.prerank_follow_author_channel import PrerankFollowAuthorChannelParitioner, PrerankFollowAuthorChannelScorer
from cascading_v2.module.channel.prerank_directly_reach_fullrank_channel import PrerankDirectlyReachFullrankChannelParitioner, PrerankDirectlyReachFullrankChannelScorer
from cascading_v2.common_module import CommonModule

class CascadingPrerankChannelSortModule(CommonModule):
  """
  添加新队列一定要添加在最后，切记！！！也不要调整已有队列的顺序，再怎么调整也不会对指标有什么正向影响！！！
  """
  _MC_QUEUES = ["photo", "picture", "follow_author", "directly_reach_fullrank"]

  def __init__(self, module_name):
    super().__init__(module_name)
  
  def process(self) -> None:
    self.flow.if_("explore_enable_skip_prerank == 0")
    scorers = self._define_scorers()
    partitioners = self._define_partitioners()
    self._weight_attr_prefix = "mc_csqw_" + self._stage() + "_"
    self._queue_definitions = [{
      "name": queue,
      "scorer": scorers[queue],
      "partitioner": partitioners[queue],
    } for queue in self._MC_QUEUES]
    self._weight_attrs = [self._get_weight_attr(queue["name"]) for queue in self._queue_definitions]
    self._score_attrs = [queue["scorer"].get_score_attr() for queue in self._queue_definitions]
    self._flag_attrs = [queue["partitioner"].get_flag_attr() for queue in self._queue_definitions]

    self.flow.gen_common_attr_by_lua(
      attr_map = {attr: "0.0" for attr in self._weight_attrs})
    
    self.flow.explore_enrich_kv_param(
      origin_param = "{{cascade_channel_sort_prerank_queue_params}}",
      param_attr_prefix = self._weight_attr_prefix,
      import_common_attr = self._weight_attrs,
      export_common_attr = self._weight_attrs,
      param_separator = ",",
      kv_separator = ":",
      param_name_list_attr = "cascade_channel_sort_prerank_queue_names",
    )
    
    for queue in self._queue_definitions:
      queue_name = queue["name"]
      weight_attr = self._get_weight_attr(queue_name)
      
      self.flow.if_(self._channel_sort_queue_enable_condition(weight_attr))
      queue["partitioner"].process()
      self.flow.end_()

    self.flow.set_attr_default_value(
      item_attrs = [{"name": attr, "type": "int", "value": 0} for attr in self._flag_attrs])

    # 这段逻辑是把 没有被任何队列标记过的item 标记到默认队列，也就是第一个队列
    self.flow.set_attr_value(
      no_overwrite = False,
      item_attrs = [
        {
          "name": self._flag_attrs[0],
          "type": "int",
          "value": 1,
        },
      ],
      select_item = { 
        "join": "and",
        "filters": [{
          "attr_name": flag_attr,
          "select_if": "==",
          "compare_to": 0,
          "select_if_attr_missing": True,
        } for flag_attr in self._flag_attrs[1:]],
      },
    )

    # 给每个队列所属的 item 打分
    # 这一步要放到所有队列的 flag 都打好之后，因为默认队列的标记需要等其他队列都打完才能上
    queue_index = 0
    for queue in self._queue_definitions:
      queue_name = queue["name"]
      weight_attr = self._get_weight_attr(queue_name)
      
      self.flow.if_(self._channel_sort_queue_enable_condition(weight_attr))
      queue["scorer"].process(self._flag_attrs[queue_index], weight_attr)
      self.flow.copy_attr(
        attrs = [
          {
            "from_item": self._score_attrs[queue_index],
            "to_item": "cascade_prerank_score",
          },
        ],
        target_item = { self._flag_attrs[queue_index]: 1 }
      )
      self.flow.end_()

      queue_index += 1
    
    self.flow.set_attr_default_value(
      item_attrs=[{"name": attr, "type": "double", "value": 0.0} for attr in self._score_attrs])

    # item attr 落盘
    self.flow._dump_attr_to_kafka(
      stage_name = "prerank_score", 
      dump_item_attr_list = [
        "mc_csqs_prerank_photo",
        "mc_csqs_prerank_picture",
        "cascade_prerank_pctr",
        "cascade_prerank_pltr",
        "is_picture",
        "is_follow_author",
        "prerank_final_index_photo",
        "explore_stat__real_show_count",
      ],
      dump_common_attr_list = [
        "active_days_avg_vv",
        "prerank_hetu_quota_control_is_degraded"
      ]
    )

    self.flow.explore_channel_sort(
      name = "explore_mc_prerank",
      traceback = True,
      channel_queue_names = "{{cascade_channel_sort_prerank_queue_names}}",
      input_count_threshold = "{{cascade_prerank_fixed_final_size}}",
      output_count = "{{cascade_prerank_fixed_final_size}}",
      weight_type = "RELATIVE",
      stage = self._stage(),
      queue_weight_attrs = self._weight_attrs,
      queue_score_attrs = self._score_attrs,
      queue_flag_attrs = self._flag_attrs,
      enable_double_lowest_score = True,
    )
    
    self.flow._perf_result(
      step_name = "prerank",
      attr_map = {
        "is_picture": ["pic", "count"],
        "is_follow_author": ["follow_author", "count"],
        "shuffle_policy": ["shuffle", "count"],
        "content_safety_level_with_namespace__level_hot_online": ["", "value_count"],
        "topk_audit_level": ["", "value_count"],
        "audit_hot_high_tag_level": ["", "value_count"],
        "audit_hot_cover_level": ["", "value_count"],
        "is_support_author_picture": ["sp_aid_pic", "count"],
        "high_value_pic_flag": ["high_value_pic", "count"],
        "is_personified_author": ["personified_author", "count"],
        "is_blacklist_author": ["blacklist_author", "count"],
        "is_hot_content": ["hot_content", "count"]
      },
      perf_sampling_attr = "_IS_PERF_SAMPLING_REQUEST_",
    )

    self.flow.log_debug_info(common_attrs=["cascade_channel_sort_prerank_queue_params", "cascade_channel_sort_prerank_queue_names"] + self._weight_attrs,
                             item_attrs=self._score_attrs + self._flag_attrs,
                             item_num_limit=10)
    self.flow.end_()
    return self

  def _channel_sort_queue_enable_condition(self, attr):
    return f"{attr} > 0.0"
  
  def _get_weight_attr(self, name):
    return f"{self._weight_attr_prefix}{name}"
  
  def _define_partitioners(self):
    partitioners = {
      "photo": PrerankPhotoChannelParitioner(self._stage() + "_photo", self.flow, self.config),
      "picture": PrerankPictureChannelParitioner(self._stage() + "_picture", self.flow, self.config),
      "follow_author": PrerankFollowAuthorChannelParitioner(self._stage() + "_follow_author", self.flow, self.config),
      "directly_reach_fullrank": PrerankDirectlyReachFullrankChannelParitioner
(self._stage() + "_directly_reach_fullrank", self.flow, self.config),
    }
    return partitioners
  
  def _define_scorers(self):
    scorers = {
      "photo": PrerankPhotoChannelScorer(self._stage() + "_photo", self.flow, self.config),
      "picture": PrerankPictureChannelScorer(self._stage() + "_picture", self.flow, self.config),
      "follow_author": PrerankFollowAuthorChannelScorer(self._stage() + "_follow_author", self.flow, self.config),
      "directly_reach_fullrank": PrerankDirectlyReachFullrankChannelScorer
(self._stage() + "_directly_reach_fullrank", self.flow, self.config),
    }
    return scorers
  
  def _stage(self):
    return "prerank"

  def _gen_photo_show_ration(self):
    self.flow \
      .set_attr_value(
        common_attrs = [
          {
            "name": "show_ration_realshow_threshold",
            "type": "int",
            "value": 10000,
          },
        ],
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "show_ration_realshow_threshold",
        ],
        import_item_attr = [
          "explore_stat__real_show_count",
          "thanos_stats__real_show_count"
        ],
        export_item_attr = [
          "show_ration_level",
        ],
        function_name = "GenPhotoShowRation",
        class_name = "ExploreLightFunctionSetV2",
      )

  def _gen_upload_time_day(self):
    self.flow \
      .enrich_attr_by_light_function(
        import_item_attr = [
          "upload_time"
        ],
        export_item_attr = [
          "upload_time_day"
        ],
        function_name = "GenUploadTimeDay",
        class_name = "ExploreLightFunctionSetV2",
      )

  def _gen_is_new_interest_explore(self):
    self.flow \
      .if_("explore_calc_new_interest_use_632 == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name" : "user_long_term_interest_cid_list", "as" : "attr_list"},
            {"name" : "explore_new_interest_max_num", "as" : "max_num_threshold"}
          ],
          import_item_attr = [
            {"name" : "cluster_id_632", "as" : "attr"}
          ],
          export_item_attr = [
            {"name" : "is_not_in_set", "as" : "is_new_interest_explore"}
          ],
          function_name = "AttrIsNotInSet",
          class_name = "ExploreLightFunctionSetV2"
        ) \
      .else_() \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name" : "uOldMmuClusterId300ListList", "as" : "attr_list"},
            {"name" : "explore_new_interest_max_num", "as" : "max_num_threshold"}
          ],
          import_item_attr = [
            {"name" : "mounted_interest_cluster_id", "as" : "attr"}
          ],
          export_item_attr = [
            {"name" : "is_not_in_set", "as" : "is_new_interest_explore"}
          ],
          function_name = "AttrIsNotInSet",
          class_name = "ExploreLightFunctionSetV2"
        ) \
      .end_()

  def calc_result_count_to_ab_metric(self):
    self._gen_photo_show_ration()
    self._gen_upload_time_day()
    self._gen_is_new_interest_explore()
    self.flow \
      .count_reco_result(
        save_count_to = "prerank_follow_author_count",
        target_item = {"is_follow_author": 1},
      ) \
      .count_reco_result(
        save_count_to = "prerank_new_interest_count",
        target_item = {"is_new_interest_explore": 1},
      ) \
      .count_reco_result(
        save_count_to = "prerank_show_ration_level6_count",
        target_item = {"show_ration_level": 6},
      ) \
      .count_reco_result(
        save_count_to = "prerank_upload_time_day0_count",
        target_item = {"upload_time_day": 0},
      ) \
      .count_reco_result(
        save_count_to = "prerank_upload_time_day1_count",
        target_item = {"upload_time_day": 1},
      ) \
      .count_reco_result(
        save_count_to = "prerank_upload_time_day2_count",
        target_item = {"upload_time_day": 2},
      ) \
      .count_reco_result(
        save_count_to = "prerank_upload_time_day3_7_count",
        target_item = {"upload_time_day": [3, 4, 5, 6, 7]},
      ) \
      .count_reco_result(
        save_count_to = "prerank_upload_time_day30_180_count",
        select_item = {
          "attr_name": "upload_time_day",
          "compare_to": 30,
          "select_if": ">=",
        } \
      ) \
      .count_reco_result(
        save_count_to = "prerank_result_count",
      ) \
      .count_reco_result(
        save_count_to = "prerank_explore_show_gt_show_ration_result_count",
        select_item = {
            "attr_name": "explore_stat__real_show_count",
            "compare_to": "{{show_ration_realshow_threshold}}",
            "select_if": ">"
        } \
      ) \
      .count_reco_result(
        save_count_to = "prerank_explore_noncoverview_result_count",
        select_item = {
          "attr_name": "audit_hot_cover_level",
          "compare_to": 0,
          "select_if": "<=",
          "select_if_attr_missing": True
        } \
      ) \
      .count_reco_result(
        save_count_to = "prerank_explore_nonsenseview_result_count",
        select_item = {
          "attr_name": "audit_b_second_tag",
          "compare_to": 0,
          "select_if": "<=",
          "select_if_attr_missing": True
        } \
      ) \
      .send_abtest_metrics(
        metrics = [
          "prerank_follow_author_count",
          "prerank_new_interest_count",
          "prerank_show_ration_level6_count",
          "prerank_upload_time_day0_count",
          "prerank_upload_time_day1_count",
          "prerank_upload_time_day2_count",
          "prerank_upload_time_day3_7_count",
          "prerank_upload_time_day30_180_count",
          "prerank_result_count",
          { "name": "is_diversity_hetu1_degraded", "as": "is_diversity_degraded" },
          "prerank_hetu_quota_control_is_degraded",
          "prerank_explore_show_gt_show_ration_result_count",
          "prerank_explore_noncoverview_result_count",
          "prerank_explore_nonsenseview_result_count"
        ],
        metric_name_prefix = "explore_reco_leaf_",
      )

  def post_process(self) -> None:
    self.flow.if_("_IS_ABTEST_METRICS_SAMPLING_REQUEST_ == 1 and _IS_ONLINE_SERVICE_ == 1 and _IS_NOT_BACKUP_ == 1")
    self.calc_result_count_to_ab_metric()
    self.flow.end_()
    self.flow \
      .if_("enable_explore_pic_cluster_counter > 0 or explore_need_traceback > 0") \
        .explore_pic_cluster_counter_enricher(
          save_pic_cluster_distr_str_attr = "prerank_pic_cluster_distr_str",
          save_long_term_interest_cnt_attr = "prerank_pic_long_term_interest_count",
          save_short_term_interest_cnt_attr = "prerank_pic_short_term_interest_count",
          save_explore_interest_cnt_attr = "prerank_pic_explore_interest_count",
          save_unknown_interest_cnt_attr = "prerank_pic_unknown_interest_count",
          save_pic_cnt_attr = "prerank_pic_count",
          save_hetu_cnt_attr = "prerank_pic_hetu_count",
          long_term_interest_list_attr = "explore_pic_long_interest_list",
          short_term_interest_list_attr = "explore_pic_short_interest_list",
          explore_interest_list_attr = "explore_pic_explore_interest_list",
          hetu_list_attr = "hetu_tag_level_info__hetu_level_one",
          target_item = {"is_picture": 1}
        ) \
      .end_()
