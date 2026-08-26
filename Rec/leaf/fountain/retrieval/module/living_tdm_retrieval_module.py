from retrieval.retrieval_module import RetrievalModule

class LivingTdmRetrievalModule(RetrievalModule):
  def __init__(self, name=str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .retrieve_common_tdm(
        name="common_living_head_tdm_explore_leaf_retr",
        kess_service="{{common_living_head_tdm_service_name_explore_leaf}}",
        traffic_type=1,
        reason=488,
        top_k="{{explore_leaf_living_head_tdm_result_num}}",
        tree_name="{{explore_leaf_living_head_tdm_tree_name}}",
        kconf_name="{{tdm_live_click_abtest_kconf_key_explore_leaf}}",
        llsid_attr_name="featureUId",  # dummy
        user_id_attr_name="featureUId",
        device_id_attr_name="featureDeviceId",
        user_info_attr_name="userInfo",
        timeout_ms=100,
        start_exp_id_attr_name="tdm_start_exp_id"
      ) \
      .limit("{{living_tdm_final_result_num_explore_leaf}}") 