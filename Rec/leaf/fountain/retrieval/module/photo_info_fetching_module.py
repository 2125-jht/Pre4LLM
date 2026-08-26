from retrieval import CommonModule

class PhotoInfoFetchingModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  # fountain_request_data_set_tags 的取值请参考 enum DataSetTrialTag 的定义
  # https://kdev.corp.kuaishou.com/git/community-science/kuaishou-ad-reco-base-proto/-/file-detail?branchName=master&filePath=src/main/proto/kuaishou/newsmodel/reco_base.proto&repoId=17967&repoName=kuaishou-ad-reco-base-proto
  # 默认值为 "1,6" 表示获取 DATA_SET_TAG_BASE 和 DATA_SET_TAG_7DAYS 数据集合
  def process(self):
    if "abtest_params_config" in self.config:
      self.flow \
        .get_abtest_params(
          **self.config["abtest_params_config"]
        )

    self.flow \
      .get_item_attr_by_distributed_flat_index(
        **self.config["distributed_flat_index_config"]
      )

    if ("additional_item_source" in self.config["distributed_flat_index_config"] and
        "common_attr" in self.config["distributed_flat_index_config"]["additional_item_source"]):
        if "fountain_retarget_interest_colossus_trigger_list" in self.config["distributed_flat_index_config"]["additional_item_source"]["common_attr"]:
          self.flow \
            .log_debug_info(
              item_list_from_attr = "fountain_retarget_interest_colossus_trigger_list",
              item_attrs = [attr["name"] if not isinstance(attr, str) else attr for attr in self.config["distributed_flat_index_config"]["attrs"]],
              for_debug_request_only = True
            )
        if "fountain_retatget_history_vv_list" in self.config["distributed_flat_index_config"]["additional_item_source"]["common_attr"]:
          self.flow \
            .log_debug_info(
              item_list_from_attr = "fountain_retatget_history_vv_list",
              item_attrs = [attr["name"] if not isinstance(attr, str) else attr for attr in self.config["distributed_flat_index_config"]["attrs"]],
              for_debug_request_only = True
            )