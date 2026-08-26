from cascading.module.queue.photo_queue import PhotoQueueParitioner
from cascading.module.queue.photo_queue import PhotoQueuePrerankScorer
from cascading.module.queue.picture_queue import PictureQueueParitioner
from cascading.module.queue.picture_queue import PictureQueuePrerankScorer
from cascading.module.queue.follow_author_queue import FollowAuthorQueueParitioner
from cascading.module.queue.follow_author_queue import FollowAuthorQueuePrerankScorer
from cascading.module.queue.white_author_queue import WhiteAuthorQueueParitioner
from cascading.module.queue.white_author_queue import WhiteAuthorQueuePrerankScorer
from cascading.module.queue.u2a_queue import U2AQueueParitioner
from cascading.module.queue.u2a_queue import U2AQueuePrerankScorer
from cascading.module.queue.directly_reach_fullrank_queue import DirectlyReachFullrankQueueParitioner
from cascading.module.queue.directly_reach_fullrank_queue import DirectlyReachFullrankQueuePrerankScorer
from cascading.common_module import CommonModule

# coding: utf-8
"""
- Description:
- Author: linpengpeng@kuaishou.com
- Date: 2022-06-16
"""

"""
添加新队列一定要添加在最后，切记！！！也不要调整已有队列的顺序，再怎么调整也不会对指标有什么正向影响！！！
"""
_MC_QUEUES = ['photo', 'picture', "follow_author", "white_author", "u2a", "directly_reach_fullrank"]

class CascadingPrerankChannelSortModule(CommonModule):

  def __init__(self, module_name):
    super().__init__(module_name)
  
  def process(self) -> None:
    self.flow.if_("explore_enable_skip_prerank == 0 and (is_zero_play_user == 0 or enable_zero_user_in_skip_prerank == 0)")
    scorers = self._define_scorers()
    partitioners = self._define_partitioners()
    self._weight_attr_prefix = 'mc_csqw_' + self._stage() + '_'
    self._queue_definitions = [{'name': queue,
                                'scorer': scorers[queue] if queue in scorers.keys() else NoopScorer(flow, f'{module_name}_{queue}', {'queue': queue}),
                                'partitioner': partitioners[queue] if queue in partitioners.keys() else NoopPartitioner(flow, f'{module_name}_{queue}', {'queue': queue})}
                                for queue in _MC_QUEUES]
    self._weight_attrs = [self._get_weight_attr(queue['name']) for queue in self._queue_definitions]
    self._score_attrs = [queue['scorer'].get_score_attr() for queue in self._queue_definitions]
    self._flag_attrs = [queue['partitioner'].get_flag_attr() for queue in self._queue_definitions]

    self.flow.gen_common_attr_by_lua(
      attr_map={attr: '0.0' for attr in self._weight_attrs})
    
    self.flow.explore_enrich_kv_param(
      origin_param='{{cascade_channel_sort_prerank_queue_params}}',
      param_attr_prefix=self._weight_attr_prefix,
      import_common_attr=self._weight_attrs,
      export_common_attr=self._weight_attrs,
      param_separator=',',
      kv_separator=':',
      param_name_list_attr='cascade_channel_sort_prerank_queue_names')
    
    for queue in self._queue_definitions:
      queue_name = queue['name']
      weight_attr = self._get_weight_attr(queue_name)
      
      self.flow.if_(self._channel_sort_queue_enable_condition(weight_attr))
      queue['partitioner'].process()
      self.flow.end_()

    self.flow.set_attr_default_value(
      item_attrs=[{'name': attr, 'type': 'int', 'value': 0} for attr in self._flag_attrs])
    

    # 这段逻辑是把 没有被任何队列标记过的item 标记到默认队列，也就是第一个队列
    statements = '+'.join([f'{attr}' for attr in self._flag_attrs[1:]])
    if len(self._flag_attrs) <= 1:
      statements = "0"
    self.flow.enrich_attr_by_lua(
      import_item_attr = self._flag_attrs[1:],
      export_item_attr=self._flag_attrs[0:1],
      function_for_item='calc',
      lua_script=f"""
      function calc(seq, item_key, reason, score)
        local s = {statements}
        if s == 0 then
          return 1
        end
        return 0
      end
      """)

    # 相当于统计各个队列的 Item 数量
    self.flow.perflog_attr_value(
      check_point='cascading_channel_sort_prerank_pre',
      item_attrs=self._flag_attrs,
      aggregator='sum')
    
    # 给每个队列所属的 item 打分
    # 这一步要放到所有队列的 flag 都打好之后，因为默认队列的标记需要等其他队列都打完才能上
    for queue in self._queue_definitions:
      queue_name = queue['name']
      weight_attr = self._get_weight_attr(queue_name)
      
      self.flow.if_(self._channel_sort_queue_enable_condition(weight_attr))
      queue['scorer'].process(queue['partitioner'].get_flag_attr(), weight_attr)
      self.flow.end_()
    
    self.flow.set_attr_default_value(
      item_attrs=[{'name': attr, 'type': 'double', 'value': 0.0} for attr in self._score_attrs])
    

    self.flow.enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "cascade_prerank_fixed_final_size", "as": "origin_size"},
          {"name": "increase_quota_status", "as": "increase_quota_status"},
          {"name": "increase_quota_after_peak_prerank_factor", "as": "factor"}
        ],
        export_common_attr = [
          {"name": "final_size", "as": "cascade_prerank_fixed_final_size"}
        ],
        function_name = "IncreaseQuotaProcess",
        class_name = "ExploreLightFunctionSetV2"
      )
    
    # item attr 落盘
    self.flow._dump_attr_to_kafka(
      stage_name = "prerank_score", 
      dump_item_attr_list = [
        "mc_csqs_prerank_photo",
        "mc_csqs_prerank_picture",
        "cascade_prerank_pctr",
        "cascade_prerank_pltr",
        "prerank_ltr",
        "prerank_ctr",
        "prerank_wtd",
        "is_picture",
        "is_long_view_author",
        "show_ration_level",
        "prerank_final_index_photo",
        "explore_stat__real_show_count",
        "explore_cold_photo_score_prerank"
      ],
      dump_common_attr_list = [
        "active_days_avg_vv",
        "prerank_hetu_quota_control_is_degraded",
        "user_risk_level"
      ]
    )

    self.flow.explore_channel_sort(
      name = "explore_mc_prerank",
      channel_queue_names='{{cascade_channel_sort_prerank_queue_names}}',
      input_count_threshold='{{cascade_prerank_fixed_final_size}}',
      output_count='{{cascade_prerank_fixed_final_size}}',
      weight_type="RELATIVE",
      stage=self._stage(),
      queue_weight_attrs=self._weight_attrs,
      queue_score_attrs=self._score_attrs,
      queue_flag_attrs=self._flag_attrs,
      enable_double_lowest_score='{{cascade_prerank_channel_sort_enable_double_lowest_score}}',
      traceback=True)
    
    # 相当于统计各个队列的 Item 数量
    self.flow.perflog_attr_value(
      check_point='cascading_channel_sort_prerank_post',
      item_attrs=self._flag_attrs,
      aggregator='sum')

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

    self.flow.log_debug_info(common_attrs=['cascade_channel_sort_prerank_queue_params', 'cascade_channel_sort_prerank_queue_names'] + self._weight_attrs,
                             item_attrs=self._score_attrs + self._flag_attrs,
                             item_num_limit=10)
    self.flow.end_()
    return self

  def _channel_sort_queue_enable_condition(self, attr):
    return f'{attr} > 0.0'
  
  def _get_weight_attr(self, name):
    return f'{self._weight_attr_prefix}{name}'
  
  def _define_partitioners(self):
    partitioners = {
      'photo': PhotoQueueParitioner(self._stage() + '_photo', self.flow, self.config),
      'picture': PictureQueueParitioner(self._stage() + '_picture', self.flow, self.config),
      'follow_author': FollowAuthorQueueParitioner(self._stage() + '_follow_author', self.flow, self.config),
      'white_author': WhiteAuthorQueueParitioner(self._stage() + '_white_author', self.flow, self.config),
      'u2a': U2AQueueParitioner(self._stage() + '_u2a', self.flow, self.config),
      'directly_reach_fullrank': DirectlyReachFullrankQueueParitioner(self._stage() + '_directly_reach_fullrank', self.flow, self.config)
   }
    return partitioners
  
  def _define_scorers(self):
    scorers = {
      'photo': PhotoQueuePrerankScorer(self._stage() + '_photo', self.flow, self.config),
      'picture': PictureQueuePrerankScorer(self._stage() + '_picture', self.flow, self.config),
      'follow_author': FollowAuthorQueuePrerankScorer(self._stage() + '_follow_author', self.flow, self.config),
      'white_author': WhiteAuthorQueuePrerankScorer(self._stage() + '_white_author', self.flow, self.config),
      'u2a': U2AQueuePrerankScorer(self._stage() + '_u2a', self.flow, self.config),
      'directly_reach_fullrank': DirectlyReachFullrankQueuePrerankScorer(self._stage() + '_directly_reach_fullrank', self.flow, self.config)
    }
    return scorers
  
  def _stage(self):
    return "prerank"

  def calc_result_count_to_ab_metric(self):
    return self.flow \
      .count_reco_result(
        save_count_to = "prerank_follow_author_count",
        target_item = {"is_follow_author": 1},
      ) \
      .count_reco_result(
        save_count_to = "prerank_all_page_valid_interest_count",
        target_item = {"is_all_page_valid_interest": 1},
      ) \
      .count_reco_result(
        save_count_to = "prerank_new_interest_count",
        target_item = {"is_new_interest_explore": 1},
      ) \
      .count_reco_result(
        save_count_to = "prerank_outer_field_interest_count",
        target_item = {"is_outer_field_interest": 1},
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
      .switch_("explore_bias_interest_type") \
        .case_(1) \
          .set_attr_value(
            no_overwrite=True,
            item_attrs=[
              {
                "name": "is_bias_interest_tagnex",
                "type": "int",
                "value": 1
              }
            ],
            select_item = { 
                "attr_name": "user_stage_interest_tagnex_tgi_score",
                "select_if": ">",
                "compare_to": 0.0,
            }
          ) \
        .case_(2) \
          .set_attr_value(
            no_overwrite=True,
            item_attrs=[
              {
                "name": "is_bias_interest_tagnex",
                "type": "int",
                "value": 1
              }
            ],
            select_item = { 
                "attr_name": "user_no_bias_interest_tagnex_tgi_score",
                "select_if": ">",
                "compare_to": 0.0,
            }
          ) \
        .case_(3) \
          .set_attr_value(
            no_overwrite=True,
            item_attrs=[
              {
                "name": "is_bias_interest_tagnex",
                "type": "int",
                "value": 1
              }
            ],
            select_item = { 
                "attr_name": "user_career_interest_tagnex_tgi_score",
                "select_if": ">",
                "compare_to": 0.0,
            }
          ) \
        .case_(4) \
          .set_attr_value(
            no_overwrite=True,
            item_attrs=[
              {
                "name": "is_bias_interest_tagnex",
                "type": "int",
                "value": 1
              }
            ],
            select_item = { 
                "attr_name": "photo_history_interest_score",
                "select_if": ">",
                "compare_to": 1.0,
            }
          ) \
        .default_() \
          .set_attr_value(
            no_overwrite=True,
            item_attrs=[
              {
                "name": "is_bias_interest_tagnex",
                "type": "int",
                "value": 1
              }
            ],
            select_item = { 
                "attr_name": "user_age_interest_tagnex_tgi_score",
                "select_if": ">",
                "compare_to": 0.0,
            }
          ) \
      .end_() \
      .count_reco_result(
        save_count_to = "prerank_bias_interest_count",
        target_item = {"is_bias_interest_tagnex": 1},
      ) \
      .send_abtest_metrics(
        metrics = [
          "prerank_bias_interest_count",
          "prerank_follow_author_count",
          "prerank_all_page_valid_interest_count",
          "prerank_new_interest_count",
          "prerank_outer_field_interest_count",
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
      .if_("enable_explore_pic_cluster_counter == 1") \
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
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "uPicLongInterestClusterIdList", "as": "long_interest_cluster_list"},
            {"name": "uPicValidInterestClusterIdList", "as": "valid_interest_cluster_list"},
            {"name": "uSingleValidPicCluster7dList", "as": "pic_single_valid_interest_cluster_list"},
            {"name": "uDoubleOutsideValidPicCluster7dList", "as": "pic_double_valid_interest_cluster_list"},
            {"name": "pic_recent_search_cluster_id_632_list", "as": "recent_search_cluster_list"},
          ],
          import_item_attr = [
            "cluster_id_632"
          ],
          export_common_attr = [
            {"name": "cluster_count", "as": "prerank_pic_cluster_count"},
            {"name": "long_interest_count", "as": "prerank_pic_long_interest_count"},
            {"name": "valid_interest_count", "as": "prerank_pic_valid_interest_count"},
            {"name": "pic_single_valid_interest_count", "as": "prerank_pic_single_valid_interest_count"},
            {"name": "pic_double_valid_interest_count", "as": "prerank_pic_double_valid_interest_count"},
            {"name": "recent_search_interest_count", "as": "prerank_pic_recent_search_interest_count"},
          ],
          function_name = "CountPicInterestClusterDistribution",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {"is_picture": 1}
        ) \
      .end_() \
      .perflog_attr_value(
        check_point = "cascading_prerank_quota_control",
        common_attrs = ["prerank_hetu_quota_control_is_degraded"],
        aggregator = "count"
      )
