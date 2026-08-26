from cascading.module.queue.photo_queue import PhotoQueueParitioner
from cascading.module.queue.photo_queue import PhotoQueueFinalScorer
from cascading.module.queue.picture_queue import PictureQueueParitioner
from cascading.module.queue.picture_queue import PictureQueueFinalScorer
from cascading.module.queue.u2a_queue import U2AQueueParitioner
from cascading.module.queue.u2a_queue import U2AQueueFinalScorer
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

class CascadingFinalChannelSortModule(CommonModule):

  def __init__(self, module_name):
    super().__init__(module_name)
    
  def process(self) -> None:
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
    
    self.flow.explore_enrich_kv_param(
      origin_param='{{final_channel_sort_queue_params_relative}}',
      param_attr_prefix=self._weight_attr_prefix,
      import_common_attr=self._weight_attrs,
      export_common_attr=self._weight_attrs,
      param_separator=',',
      kv_separator=':',
      param_name_list_attr='final_channel_sort_queue_names')

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
        "sum_of_all_weight_attrs": sum_weight_statements,
        "final_channel_sort_use_relative_weight": "1"
      }
    )
    
    # 相当于统计各个队列的 Item 数量
    self.flow.perflog_attr_value(
      check_point='final_channel_sort_pre',
      item_attrs=self._flag_attrs,
      aggregator='sum')

    self.flow.enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "mc_final_candidate_num", "as": "origin_size"},
          {"name": "increase_quota_status", "as": "increase_quota_status"},
          {"name": "increase_quota_after_peak_mc_factor", "as": "factor"}
        ],
        export_common_attr = [
          {"name": "final_size", "as": "mc_final_candidate_num"}
        ],
        function_name = "IncreaseQuotaProcess",
        class_name = "ExploreLightFunctionSetV2"
      )
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
          {"name": "final_channel_sort_use_relative_weight", "as": "use_relative_weight"},
          {"name": "sum_of_all_weight_attrs", "as": "sum_of_all_weight"},
          {"name": "mc_final_candidate_num", "as": "sum_of_all_channel_target_count"}
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
        common_attrs=[absolute_weight_attr, weight_attr, left_count_attr, origin_count_attr, 'sum_of_all_weight_attrs', 'mc_final_candidate_num', 'final_channel_sort_use_relative_weight'],
        for_debug_request_only = True
      )  
    
    self.flow.set_attr_default_value(
      item_attrs=[{'name': attr, 'type': 'double', 'value': 0.0} for attr in self._score_attrs])

    # item attr 落盘
    self.flow._dump_attr_to_kafka(
      stage_name = "mc_s2_score", 
      dump_item_attr_list = [
        "mc_csqs_cascade_stage2_photo",
        "mc_csqs_cascade_stage2_picture",
        "cascade_dstill_pctr",
        "cascade_dstill_pltr",
        "cascade_dstill_plvtr",
        "cascade_dstill_pwatch_time"
      # 可再添加 mc2 使用而 mc1 未使用 ( eg: 全连接分 )的新字段
      ]
    )

    self.flow.if_("explore_mc_ensemble_s2_skip_truncate == 0")
    self.flow.explore_channel_sort(
      name = "explore_mc_stage2",
      channel_queue_names='{{final_channel_sort_queue_names}}',
      weight_type="RELATIVE",
      output_count='{{mc_final_candidate_num}}',
      stage=self._stage(),
      queue_weight_attrs=self._weight_attrs,
      queue_score_attrs=self._score_attrs,
      queue_flag_attrs=self._flag_attrs,
      enable_double_lowest_score='{{cascade_final_channel_sort_enable_double_lowest_score}}',
      traceback=True)
    self.flow.end_()
    
    # 相当于统计各个队列的 Item 数量
    self.flow.perflog_attr_value(
      check_point='final_channel_sort_post',
      item_attrs=self._flag_attrs,
      aggregator='sum')
    
    
    self.flow.log_debug_info(common_attrs=['final_channel_sort_queue_params', 'final_channel_sort_queue_names'] + self._weight_attrs,
                             item_attrs=self._score_attrs + self._flag_attrs,
                             item_num_limit=10)

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
      'u2a': U2AQueueParitioner(self._stage() + '_u2a', self.flow, self.config),
   }
    return partitioners
  
  def _define_scorers(self):
    scorers = {
      'photo': PhotoQueueFinalScorer(self._stage() + '_photo', self.flow, self.config),
      'picture': PictureQueueFinalScorer(self._stage() + '_picture', self.flow, self.config),
      'u2a': U2AQueueFinalScorer(self._stage() + '_u2a', self.flow, self.config),
    }
    return scorers

  def _stage(self):
    return "cascade_stage2"


  def post_process(self) -> None:
    # 统计结果量级
    self.flow.count_reco_result(
        save_count_to = "cascade_s2_result_count"
      ) \
      .count_reco_result(
        save_count_to = "cascade_s2_pic_result_count",
        target_item = {"is_picture": 1}
      ) \
      .send_abtest_metrics(
        metrics = [
          {"name": "cascade_s2_result_count", "as": "explore_reco_leaf_cascade_s2_result_count"},
          {"name": "cascade_s2_pic_result_count", "as": "explore_reco_leaf_cascade_s2_pic_count"},
        ],
        metric_name_prefix = "",
      )
