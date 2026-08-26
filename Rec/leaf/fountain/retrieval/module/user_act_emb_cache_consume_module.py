from retrieval.retrieval_module import RetrievalModule

# 不是一个真正的召回，特殊需求，不要参考
class UserActEmbCacheConsumeModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self):
    self.flow \
      .if_("enable_fountain_user_rank_real_act_cache == 1 or enable_fountain_user_full_rank_real_act_cache == 1") \
        .gen_common_attr_by_lua(
          attr_map={
            "act_user_id_or_device_id": "_USER_ID_ == 0 and util.CityHash64(_DEVICE_ID_) or _USER_ID_",
          }
        ) \
        .if_("enable_fountain_user_rank_real_act_cache == 1") \
          .delegate_enrich(
            kess_service="{{fountain_user_rank_real_act_cache_model_kess_name}}",
            send_common_attrs=["act_user_id_or_device_id"],
            consistent_hash=True,
            hash_id="{{act_user_id_or_device_id}}",
            request_type="predict_cache_for_gpt",
            timeout_ms=50,
            recv_common_attrs=[{"name":"act_user_id_or_device_id", "as": "act_res"}]
          ) \
        .end_if_() \
        .if_("enable_fountain_user_full_rank_real_act_cache == 1") \
          .if_("enable_fountain_user_rank_real_act_cache == 0 or {{fountain_user_full_rank_real_act_cache_model_kess_name}} ~= {{fountain_user_rank_real_act_cache_model_kess_name}}") \
            .delegate_enrich(
              kess_service="{{fountain_user_full_rank_real_act_cache_model_kess_name}}",
              send_common_attrs=["act_user_id_or_device_id"],
              consistent_hash=True,
              hash_id="{{act_user_id_or_device_id}}",
              request_type="predict_cache_for_gpt",
              timeout_ms=50,
              recv_common_attrs=[{"name":"act_user_id_or_device_id", "as": "act_res"}]
            ) \
          .end_if_() \
        .end_if_() \
      .end_if_()