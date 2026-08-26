"""
需要将 dragon 目录加入到下面环境变量中（可以添加到类似 ~/.bashrc 中每次启动自动配置环境变量）
export PYTHONPATH={GIT_ROOT_PATH}:$PYTHONPATH
"""
import sys
sys.path.append("./degrading")


from dragonfly.common_leaf_dsl import LeafService
from degrading import ExploreDiversifyingFlow
from degrading import FountainCorrelationFlow
from degrading import FountainDiversifyingFlow

explore_diversifing_flow = ExploreDiversifyingFlow("explore_diversifing_flow")
fountain_correlation_flow = FountainCorrelationFlow("fountain_correlation_flow")
fountain_diversifing_flow = FountainDiversifyingFlow("fountain_diversifing_flow")

service = LeafService(
  kess_name = "grpc_exploreDegradingLeafDryrun",
  common_attrs_from_request = [
    "page_size",
    "source_hetu_v2_face_id_list",
    "source_hetu_v2_tag_list",
    "source_hetu_v2_level_one_list",
    "source_hetu_v2_level_two_list",
    "source_hetu_v2_level_three_list",
  ],
)
service \
  .return_item_attrs([
    { "name": "author__id", "as": "aId" },
    { "name": "hetu_tag_level_info__hetu_level_one", "as": "hetu_level_one_list" },
    { "name": "hetu_tag_level_info__hetu_level_two", "as": "hetu_level_two_list" },
    { "name": "author_age_info__age_segment", "as": "age_segment" },
    { "name": "empirical_ltr", "as": "pltr" },
    { "name": "empirical_ftr", "as": "pftr" },
    { "name": "empirical_wtr", "as": "pwtf" },
    { "name": "empirical_htr", "as": "phtr" },
    { "name": "empirical_wtd", "as": "pwtd" },
  ]) \
  .add_leaf_flows([explore_diversifing_flow], request_type = "explore_diversifing", as_default = True) \
  .add_leaf_flows([fountain_correlation_flow], request_type = "fountain_correlation") \
  .add_leaf_flows([fountain_diversifing_flow], request_type = "fountain_diversifing") \
  .build(output_file = "dynamic_json_config.json")
