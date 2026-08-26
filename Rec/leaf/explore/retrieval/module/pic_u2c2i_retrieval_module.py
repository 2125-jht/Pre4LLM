from retrieval import RetrievalModule

class PicU2C2IRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .set_attr_value(
        no_overwrite = True,
        common_attrs = [
          {
            "name": "trigger_cids",
            "type": "int_list",
            "value": []
          }
        ]
      )
    # 视频兴趣 cluster
    self.flow.if_("explore_pic_u2c2i_retr_enable_valid_interest_cluster == 1")
    self._add_triggers("uPicValidInterestClusterIdList", "explore_pic_u2c2i_retr_valid_interest_cluster_trigger_num")
    self.flow.end_()
    self.flow.if_("explore_pic_u2c2i_retr_enable_long_interest_cluster == 1")
    self._add_triggers("uPicLongInterestClusterIdList", "explore_pic_u2c2i_retr_long_interest_cluster_trigger_num")
    self.flow.end_()
    self.flow.if_("explore_pic_u2c2i_retr_enable_search_interest_cluster == 1")
    self._add_triggers("uPicSearchInterestClusterIdList", "explore_pic_u2c2i_retr_search_interest_cluster_trigger_num")
    self.flow.end_()
    # 图文兴趣 cluster
    self.flow.if_("explore_pic_u2c2i_retr_enable_double_pic_valid_interest_cluster == 1 and (uDoubleOutsideValidPicClusterCnt7dKV or 0) < explore_pic_u2c2i_retr_user_pic_cluster_thresh")
    self._add_triggers("uDoubleOutsideValidPicCluster7dList", "explore_pic_u2c2i_retr_double_pic_valid_interest_cluster_trigger_num")
    self.flow.end_()
    self.flow.if_("explore_pic_u2c2i_retr_enable_single_pic_valid_interest_cluster == 1 and (uDoubleOutsideValidPicClusterCnt7dKV or 0) < explore_pic_u2c2i_retr_user_pic_cluster_thresh")
    self._add_triggers("uSingleValidPicCluster7dList", "explore_pic_u2c2i_retr_single_pic_valid_interest_cluster_trigger_num")
    self.flow.end_()
    self.flow.if_("explore_pic_u2c2i_retr_enable_double_pic_growth_interest_cluster == 1 and (uDoubleOutsideValidPicClusterCnt7dKV or 0) < explore_pic_u2c2i_retr_user_pic_cluster_thresh")
    self._add_triggers("uPicGrowthCidList", "explore_pic_u2c2i_retr_double_pic_growth_interest_cluster_trigger_num")
    self.flow.end_()
    self.flow \
      .shuffle_list_attr(
        common_attr = "trigger_cids"
      ) \
      .truncate(
        item_list_from_attr = "trigger_cids",
        size_limit = "{{explore_pic_u2c2i_retr_trigger_num}}",
      ) \
      .retrieve_by_redis(
        reason = self.reason,
        retrieve_num = "{{explore_pic_u2c2i_retr_num}}",
        retrieve_num_per_key = "{{explore_pic_u2c2i_retr_num_per_key}}",
        cluster_name = "explorePicCache",
        timeout_ms = 50,
        key_from_attr = "trigger_cids",
        key_prefix = "{{explore_pic_u2c2i_redis_key_prefix}}",
        item_separator = ","
      ) \
      .deduplicate() \
      .shuffle() \
      .limit(size = "{{explore_pic_u2c2i_retr_num_final}}")

  def _add_triggers(self, cid_list_attr: str, trigger_num_attr: str):
    self.flow \
      .copy_attr(
        attrs = [{
          "from_common": cid_list_attr,
          "to_common": "tmp_cids"
        }]
      ) \
      .shuffle_list_attr(
        common_attr = "tmp_cids"
      ) \
      .pack_common_attr(
        input_common_attrs = [
          "trigger_cids",
          "tmp_cids",
        ],
        output_common_attr = "trigger_cids",
        deduplicate = True,
        limit_num = "{{return #trigger_cids + " + trigger_num_attr + "}}"
      )
