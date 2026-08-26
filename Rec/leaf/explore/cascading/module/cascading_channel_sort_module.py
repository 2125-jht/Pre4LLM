from cascading.module.queue.photo_queue import PhotoQueueParitioner
from cascading.module.queue.photo_queue import PhotoQueueCascadingScorer
from cascading.module.queue.picture_queue import PictureQueueParitioner
from cascading.module.queue.picture_queue import PictureQueueCascadingScorer
from cascading.module.queue.u2a_queue import U2AQueueParitioner
from cascading.module.queue.u2a_queue import U2AQueueCascadingScorer
from cascading.module.queue.picture_queue import PictureQueueCascadingScorer
from cascading.module.queue.directly_reach_fullrank_queue import DirectlyReachFullrankQueueParitioner
from cascading.module.queue.directly_reach_fullrank_queue import DirectlyReachFullrankQueueCascadingScorer
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
_MC_QUEUES = ['photo', 'picture', "u2a", "directly_reach_fullrank"]

class CascadingChannelSortModule(CommonModule):

  def __init__(self, module_name):
    super().__init__(module_name)
    
  def process(self) -> None:
    self.flow.if_("explore_enable_skip_cascade_s1 == 0 and (is_zero_play_user == 0 or explore_enable_zero_user_skip_cascade_s1 == 0)")
    enable_attrs = self._define_enable_attrs()
    scorers = self._define_scorers()
    partitioners = self._define_partitioners()
    self._weight_attr_prefix = 'mc_csqw_' + self._stage() + '_'
    self._absolute_weight_attr_prefix = 'mc_csqaw_' + self._stage() + '_'
    self._left_count_attr_prefix = 'mc_csqlc_' + self._stage() + '_'
    self._origin_count_attr_prefix = 'mc_csqoc_' + self._stage() + '_'
    self._queue_definitions = [{'name': queue,
                                'scorer': scorers[queue] if queue in scorers.keys() else NoopScorer(flow, f'{module_name}_{queue}', {'queue': queue}),
                                'partitioner': partitioners[queue] if queue in partitioners.keys() else NoopPartitioner(flow, f'{module_name}_{queue}', {'queue': queue})}
                                for queue in _MC_QUEUES]
    self._weight_attrs = [self._get_weight_attr(queue['name']) for queue in self._queue_definitions]
    self._score_attrs = [queue['scorer'].get_score_attr() for queue in self._queue_definitions]
    self._flag_attrs = [queue['partitioner'].get_flag_attr() for queue in self._queue_definitions]

    self.flow.gen_common_attr_by_lua(
      attr_map={attr: '0.0' for attr in self._weight_attrs})
    
    self.flow.if_('cascade_channel_sort_use_relative_weight == 1')
    self.flow.explore_enrich_kv_param(
      origin_param='{{cascade_channel_sort_queue_params_relative}}',
      param_attr_prefix=self._weight_attr_prefix,
      import_common_attr=self._weight_attrs,
      export_common_attr=self._weight_attrs,
      param_separator=',',
      kv_separator=':',
      param_name_list_attr='cascade_channel_sort_queue_names')
    self.flow.else_()
    self.flow.explore_enrich_kv_param(
      origin_param='{{cascade_channel_sort_queue_params}}',
      param_attr_prefix=self._weight_attr_prefix,
      import_common_attr=self._weight_attrs,
      export_common_attr=self._weight_attrs,
      param_separator=',',
      kv_separator=':',
      param_name_list_attr='cascade_channel_sort_queue_names')
    self.flow.end_()

    # 给每个队列所属的 item 打标签
    for queue in self._queue_definitions:
      queue_name = queue['name']
      enable_attr = enable_attrs[queue_name]
      weight_attr = self._get_weight_attr(queue_name)
      
      self.flow.if_(self._channel_sort_queue_enable_condition(enable_attr, weight_attr))
      queue['partitioner'].process()
      self.flow.end_()

    self.flow.set_attr_default_value(
      item_attrs=[{'name': attr, 'type': 'int', 'value': 0} for attr in self._flag_attrs])
    
    # 这段逻辑是把 没有被任何队列标记过的item 标记到默认队列，也就是第一个队列
    statements = '+'.join([f'{attr}' for attr in self._flag_attrs[1:]])
    if len(self._flag_attrs) <= 1:
      statements = "0"
    self.flow.enrich_attr_by_lua(
      import_item_attr=self._flag_attrs[1:],
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

    # 这段逻辑是统计 weight_attrs 的总和，用于之后计算每个 channel 最后留下多少个 item
    sum_weight_statements = '+'.join([f'{attr}' for attr in self._weight_attrs])
    self.flow.gen_common_attr_by_lua(
      attr_map={
        "sum_of_all_weight_attrs": sum_weight_statements
      }
    )
    
    # 相当于统计各个队列的 Item 数量
    self.flow.perflog_attr_value(
      check_point='cascading_channel_sort_pre',
      item_attrs=self._flag_attrs,
      aggregator='sum')

    # 给每个队列所属的 item 打分
    # 这一步要放到所有队列的 flag 都打好之后，因为默认队列的标记需要等其他队列都打完才能上
    
    for queue in self._queue_definitions:
      queue_name = queue['name']
      enable_attr = enable_attrs[queue_name]
      weight_attr = self._get_weight_attr(queue_name)
      origin_count_attr = self._get_origin_count_attr(queue_name)
      left_count_attr = self._get_left_count_attr(queue_name)
      absolute_weight_attr = self._get_absolute_weight_attr(queue_name)
      
      self.flow.if_(self._channel_sort_queue_enable_condition(enable_attr, weight_attr))
      
      self.flow.count_reco_result(
        save_count_to = origin_count_attr,
        target_item = {queue['partitioner'].get_flag_attr(): 1}
      )

      self.flow.enrich_attr_by_light_function(
        import_common_attr = [
          {"name": origin_count_attr, "as": "origin_count"},
          {"name": weight_attr, "as": "weight"},
          {"name": "cascade_channel_sort_use_relative_weight", "as": "use_relative_weight"},
          {"name": "sum_of_all_weight_attrs", "as": "sum_of_all_weight"},
          {"name": "cascade_channel_sort_stage1_fixed_final_size", "as": "sum_of_all_channel_target_count"}
        ],
        export_common_attr = [
          {"name": "left_count", "as": left_count_attr},
          {"name": "absolute_weight", "as": absolute_weight_attr}
        ],
        function_name = "CalcLeftItemCount",
        class_name = "ExploreLightFunctionSetV2",
      )

      queue['scorer'].process(queue['partitioner'].get_flag_attr(), absolute_weight_attr, left_count_attr)
      self.flow.end_()

      self.flow.log_debug_info(
        common_attrs=[absolute_weight_attr, weight_attr, left_count_attr, origin_count_attr, 'sum_of_all_weight_attrs', 'cascade_channel_sort_stage1_fixed_final_size'],
        for_debug_request_only = True
      )  
    
    self.flow.set_attr_default_value(
      item_attrs=[{'name': attr, 'type': 'double', 'value': 0.0} for attr in self._score_attrs])

    # item attr 落盘
    self.flow._dump_attr_to_kafka(
      stage_name = "mc_s1_score", 
      dump_item_attr_list = [
        "mc_csqs_cascade_stage1_photo",
        "mc_csqs_cascade_stage1_picture",
        "cascade_cluster_id",
        # ES 队列
        "cascade_score",
        "mc_ensemble_pwatch_time",
        "cascade_pwtd_inverse",
        "mc_ensemble_plvtr",
        "mc_ensemble_plvtr2",
        "mc_ensemble_pctr",
        "mc_ensemble_pltr",
        "mc_ensemble_pwtr",
        "mc_ensemble_pftr",
        "mc_ensemble_ptr",
        "mc_ensemble_pepstr",
        "mc_ensemble_pcmtr",
        "mc_ensemble_pcltr",
        "cascade_phtr",
        "mc_ensemble_psvtr",
        "mc_ensemble_smooth_age_score",
        "mc_ensemble_peftr",
        "mc_ensemble_pefctr",
        "mc_ensemble_pwtd_inverse",
        "mc_ensemble_pfptr",
        # emp xtr
        "empirical_ctr",
        "empirical_ftr",
        "empirical_htr",
        "empirical_ltr",
        "empirical_lvtr",
        "empirical_ptr",
        "empirical_svtr",
        "empirical_wtr",
      ],
      dump_common_attr_list = [
        "dynamic_pic_quota",
        "external_prefer_user_flag"
      ]
    )

    self.flow.if_('cascade_channel_sort_use_relative_weight == 1')
    self.flow.explore_channel_sort(
      name = "explore_mc_stage1_relative",
      channel_queue_names='{{cascade_channel_sort_queue_names}}',
      weight_type="RELATIVE",
      output_count='{{cascade_channel_sort_stage1_fixed_final_size}}',
      stage=self._stage(),
      queue_weight_attrs=self._weight_attrs,
      queue_score_attrs=self._score_attrs,
      queue_flag_attrs=self._flag_attrs,
      enable_double_lowest_score='{{cascade_channel_sort_enable_double_lowest_score}}',
      traceback=True)
    self.flow.else_()
    self.flow.explore_channel_sort(
      name = "explore_mc_stage1_absoulte",
      channel_queue_names='{{cascade_channel_sort_queue_names}}',
      weight_type="ABSOLUTE",
      stage=self._stage(),
      queue_weight_attrs=self._weight_attrs,
      queue_score_attrs=self._score_attrs,
      queue_flag_attrs=self._flag_attrs,
      enable_double_lowest_score='{{cascade_channel_sort_enable_double_lowest_score}}',
      traceback=True)
    self.flow.end_()
    # 相当于统计各个队列的 Item 数量
    self.flow.perflog_attr_value(
      check_point='cascading_channel_sort_post',
      item_attrs=self._flag_attrs,
      aggregator='sum')

    self.flow._perf_result(
      step_name = "stage1",
      attr_map = {
        "is_picture": ["pic", "count"],
        "is_follow_author": ["follow_author", "count"],
        "shuffle_policy": ["shuffle", "count"],
        "content_safety_level_with_namespace__level_hot_online": ["", "value_count"],
        "topk_audit_level": ["", "value_count"],
        "audit_hot_high_tag_level": ["", "value_count"],
        "audit_hot_cover_level": ["", "value_count"],
        "audit_b_second_tag": ["", "value_count"],
        "is_support_author_picture": ["sp_aid_pic", "count"],
        "high_value_pic_flag": ["high_value_pic", "count"],
        "is_explore_photo": ["explore", "count"],
        "is_high_quality_explore_photo": ["high_quality_explore", "count"]
      },
      perf_sampling_attr = "_IS_PERF_SAMPLING_REQUEST_",
    )
    
    self.flow.log_debug_info(common_attrs=['cascade_channel_sort_queue_params', 'cascade_channel_sort_queue_names'] + self._weight_attrs,
                             item_attrs=self._score_attrs + self._flag_attrs,
                             item_num_limit=10)
    self.flow.end_()

  def _channel_sort_queue_enable_condition(self, enable_attr, weight_attr):
    return f'{enable_attr} == 1 and {weight_attr} > 0.0'

  def _get_weight_attr(self, name):
    return f'{self._weight_attr_prefix}{name}'
  
  def _get_absolute_weight_attr(self, name):
    return f'{self._absolute_weight_attr_prefix}{name}'
  
  def _get_origin_count_attr(self, name):
    return f'{self._origin_count_attr_prefix}{name}'

  def _get_left_count_attr(self, name):
    return f'{self._left_count_attr_prefix}{name}'

  def _define_enable_attrs(self):
    attr_pattern = "enable_explore_" + self._stage() + "_{0}_channel"
    enable_attrs = {
      "photo": attr_pattern.format("photo"),
      "picture": attr_pattern.format("picture"),
      "u2a": attr_pattern.format("u2a"),
      "directly_reach_fullrank": attr_pattern.format("directly_reach_fullrank"),
    }
    return enable_attrs

  def _define_partitioners(self):
    partitioners = {
      'photo': PhotoQueueParitioner(self._stage() + '_photo', self.flow, self.config),
      'picture': PictureQueueParitioner(self._stage() + '_picture', self.flow, self.config),
      'u2a': U2AQueueParitioner(self._stage() + '_u2a', self.flow, self.config),
      'directly_reach_fullrank': DirectlyReachFullrankQueueParitioner(self._stage() + '_directly_reach_fullrank', self.flow, self.config)
   }
    return partitioners
  
  def _define_scorers(self):
    scorers = {
      'photo': PhotoQueueCascadingScorer(self._stage() + '_photo', self.flow, self.config),
      'picture': PictureQueueCascadingScorer(self._stage() + '_picture', self.flow, self.config),
      'u2a': U2AQueueCascadingScorer(self._stage() + '_u2a', self.flow, self.config),
      'directly_reach_fullrank': DirectlyReachFullrankQueueCascadingScorer(self._stage() + '_directly_reach_fullrank', self.flow, self.config)
    }
    return scorers

  def _stage(self):
    return "cascade_stage1"

  def calc_result_count_to_ab_metric(self):
    return self.flow \
      .count_reco_result(
        save_count_to = "cascade_s1_all_page_valid_interest_count",
        target_item = {"is_all_page_valid_interest": 1},
      ) \
      .count_reco_result(
        save_count_to = "cascade_s1_new_interest_count",
        target_item = {"is_new_interest_explore": 1},
      ) \
      .count_reco_result(
        save_count_to = "cascade_s1_outer_field_interest_count",
        target_item = {"is_outer_field_interest": 1},
      ) \
      .count_reco_result(
        save_count_to = "cascade_s1_show_ration_level6_count",
        target_item = {"show_ration_level": 6},
      ) \
      .count_reco_result(
        save_count_to = "cascade_s1_upload_time_day0_count",
        target_item = {"upload_time_day": 0},
      ) \
      .count_reco_result(
        save_count_to = "cascade_s1_upload_time_day1_count",
        target_item = {"upload_time_day": 1},
      ) \
      .count_reco_result(
        save_count_to = "cascade_s1_upload_time_day2_count",
        target_item = {"upload_time_day": 2},
      ) \
      .count_reco_result(
        save_count_to = "cascade_s1_upload_time_day3_7_count",
        target_item = {"upload_time_day": [3, 4, 5, 6, 7]},
      ) \
      .count_reco_result(
        save_count_to = "cascade_s1_upload_time_day30_180_count",
        select_item = {
          "attr_name": "upload_time_day",
          "compare_to": 30,
          "select_if": ">=",
        } \
      ) \
      .count_reco_result(
        save_count_to = "cascade_s1_result_count",
      ) \
      .count_reco_result(
        save_count_to = "cascade_s1_explore_show_gt_show_ration_result_count",
        select_item = {
            "attr_name": "explore_stat__real_show_count",
            "compare_to": "{{show_ration_realshow_threshold}}",
            "select_if": ">"
        } \
      ) \
      .count_reco_result(
        save_count_to = "cascade_s1_explore_noncoverview_result_count",
        select_item = {
          "attr_name": "audit_hot_cover_level",
          "compare_to": 0,
          "select_if": "<=",
          "select_if_attr_missing": True
        } \
      ) \
      .count_reco_result(
        save_count_to = "cascade_s1_explore_nonsenseview_result_count",
        select_item = {
          "attr_name": "audit_b_second_tag",
          "compare_to": 0,
          "select_if": "<=",
          "select_if_attr_missing": True
        } \
      ) \
      .count_reco_result(
        save_count_to = "cascade_s1_bias_interest_count",
        target_item = {"is_bias_interest_tagnex": 1},
      ) \
      .send_abtest_metrics(
        metrics = [
          "cascade_s1_bias_interest_count",
          "cascade_s1_all_page_valid_interest_count",
          "cascade_s1_new_interest_count",
          "cascade_s1_outer_field_interest_count",
          "cascade_s1_show_ration_level6_count",
          "cascade_s1_upload_time_day0_count",
          "cascade_s1_upload_time_day1_count",
          "cascade_s1_upload_time_day2_count",
          "cascade_s1_upload_time_day3_7_count",
          "cascade_s1_upload_time_day30_180_count",
          "cascade_s1_result_count",
          "cascade_s1_explore_show_gt_show_ration_result_count",
          "cascade_s1_explore_noncoverview_result_count",
          "cascade_s1_explore_nonsenseview_result_count"
        ],
        metric_name_prefix = "explore_reco_leaf_",
      )

  def post_process(self) -> None:
    self.flow.if_("_IS_ABTEST_METRICS_SAMPLING_REQUEST_ == 1 and _IS_ONLINE_SERVICE_ == 1 and _IS_NOT_BACKUP_ == 1")
    self.calc_result_count_to_ab_metric()
    self.flow.end_()
    self.flow \
    .pack_item_attr(  # 保存粗排 s1 结束后的结果集
      item_source = {
        "reco_results": True
      },
      mappings = [{
        "aggregator": "concat",
        "from_item_attr": "item_key",
        "to_common_attr": "cascade_output_item_key_list"
      }],
    ) \
    .if_("enable_cascade_channel_caption_boost == 1") \
      .perflog_attr_value(
        check_point = "cascade_channel_caption",
        common_attrs = [
          "cascade_channel_caption_photo_boost_count"
        ],
      ) \
    .end_() \
    .if_("enable_explore_pic_cluster_counter == 1") \
      .explore_pic_cluster_counter_enricher(
        save_pic_cluster_distr_str_attr = "mc_s1_pic_cluster_distr_str",
        save_long_term_interest_cnt_attr = "mc_s1_pic_long_term_interest_count",
        save_short_term_interest_cnt_attr = "mc_s1_pic_short_term_interest_count",
        save_explore_interest_cnt_attr = "mc_s1_pic_explore_interest_count",
        save_unknown_interest_cnt_attr = "mc_s1_pic_unknown_interest_count",
        save_pic_cnt_attr = "mc_s1_pic_count",
        save_hetu_cnt_attr = "mc_s1_pic_hetu_count",
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
          {"name": "cluster_count", "as": "mc_s1_pic_cluster_count"},
          {"name": "long_interest_count", "as": "mc_s1_pic_long_interest_count"},
          {"name": "valid_interest_count", "as": "mc_s1_pic_valid_interest_count"},
          {"name": "pic_single_valid_interest_count", "as": "mc_s1_pic_single_valid_interest_count"},
          {"name": "pic_double_valid_interest_count", "as": "mc_s1_pic_double_valid_interest_count"},
          {"name": "recent_search_interest_count", "as": "mc_s1_pic_recent_search_interest_count"},
        ],
        function_name = "CountPicInterestClusterDistribution",
        class_name = "ExploreLightFunctionSetV2",
        target_item = {"is_picture": 1}
      ) \
    .end_()
