# coding: utf-8
"""
- Description:
- Author: linpengpeng@kuaishou.com
- Date: 2022-06-17
"""
from abc import abstractmethod
from cascading.common_module import CommonModule


class ChannelSortQueuePartitioner:
  """
  Cascading 阶段 Channel Sort 中指定某个 Item 是否属于该队列
  _flag_attr: 表示该 Item 是否属于该队列，框架默认赋值 0，如果某个 Item 属于该队列对其赋值 1
  """
  
  def __init__(self, name, flow, config):
    self._flow = flow
    self._config = config
    self._flag_attr = f'mc_csqf_{name}' # csqf -> Channel_Sort_Queue_Flag 

  @property
  def flow(self):
    return self._flow

  def process(self):
    self._calculate_flag()
  
  def get_flag_attr(self):
    return self._flag_attr

  def _enrich(self):
    self._calculate_flag()
  
  def _get_item_attrs(self):
    return [self._flag_attr]

  @abstractmethod
  def _calculate_flag(self):
    pass

class NoopPartitioner(ChannelSortQueuePartitioner):

  def _calculate_flag(self):
    pass

class ChannelSortQueueScorer:
  """
  子类重载 _caculate_score 方法就行了，其他的不要动，flag_attr(item_attr) 为 1 的是需要打分的 item 
  _score_attr: 给 Item 的打分
  """
  def __init__(self, name, flow, config):
    self._flow = flow
    self._config = config
    self._score_attr = f'mc_csqs_{name}' #  csqs -> Channel_Sort_Queue_Score

  @property
  def flow(self):
    return self._flow

  def process(self, flag_attr, weight_attr, left_count_attr=None):
    self._caculate_score(flag_attr, weight_attr, left_count_attr)

  
  def get_score_attr(self):
    return self._score_attr
  
  def _get_score_attrs(self):
    return [self._score_attr]

  @abstractmethod
  def _caculate_score(self, flag_attr, weight_attr, left_count_attr):
    pass

class NoopScorer(ChannelSortQueueScorer):

  def _caculate_score(self, flag_attr, weight_attr, left_count_attr):
    pass
