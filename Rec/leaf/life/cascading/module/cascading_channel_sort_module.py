from cascading.module.queue.photo_queue import PhotoQueueParitioner
from cascading.module.queue.photo_queue import PhotoQueueCascadingScorer
from cascading.module.queue.picture_queue import PictureQueueParitioner
from cascading.module.queue.picture_queue import PictureQueueCascadingScorer
from cascading.module.queue.u2a_queue import U2AQueueParitioner
from cascading.module.queue.u2a_queue import U2AQueueCascadingScorer
from cascading.module.queue.picture_queue import PictureQueueCascadingScorer
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
_MC_QUEUES = ['photo', 'picture', "u2a"]

class CascadingChannelSortModule(CommonModule):

  def __init__(self, module_name):
    super().__init__(module_name)
    
  def process(self) -> None:
    self.flow.if_("explore_enable_skip_cascade_s1 == 0")
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
      weight_attr = self._get_weight_attr(queue_name)
      origin_count_attr = self._get_origin_count_attr(queue_name)
      left_count_attr = self._get_left_count_attr(queue_name)
      absolute_weight_attr = self._get_absolute_weight_attr(queue_name)
      
      self.flow.if_(self._channel_sort_queue_enable_condition(weight_attr))
      
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

  def _channel_sort_queue_enable_condition(self, attr):
    return f'{attr} > 0.0'
  
  def _get_weight_attr(self, name):
    return f'{self._weight_attr_prefix}{name}'
  
  def _get_absolute_weight_attr(self, name):
    return f'{self._absolute_weight_attr_prefix}{name}'
  
  def _get_origin_count_attr(self, name):
    return f'{self._origin_count_attr_prefix}{name}'

  def _get_left_count_attr(self, name):
    return f'{self._left_count_attr_prefix}{name}'

  def _define_partitioners(self):
    partitioners = {
      'photo': PhotoQueueParitioner(self._stage() + '_photo', self.flow, self.config),
      'picture': PictureQueueParitioner(self._stage() + '_picture', self.flow, self.config),
      'u2a': U2AQueueParitioner(self._stage() + '_u2a', self.flow, self.config) 
   }
    return partitioners
  
  def _define_scorers(self):
    scorers = {
      'photo': PhotoQueueCascadingScorer(self._stage() + '_photo', self.flow, self.config),
      'picture': PictureQueueCascadingScorer(self._stage() + '_picture', self.flow, self.config),
      'u2a': U2AQueueCascadingScorer(self._stage() + '_u2a', self.flow, self.config) 
    }
    return scorers

  def _stage(self):
    return "cascade_stage1"


  def post_process(self) -> None:
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
    .send_abtest_metrics(
      metrics = [
        {"name": "cascading_stage1_result_count", "as": "explore_reco_leaf_cascade_s1_result_count"},
        {"name": "cascading_stage1_pic_result_count", "as": "explore_reco_leaf_cascade_s1_pic_count"},
      ],
      metric_name_prefix = "",
    )
