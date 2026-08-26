from cascading import CommonModule

class CascadingUaLtvModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    """cascade explore ua ltv model"""
    self.flow \
    .if_("enable_explore_cascade_follow_ltv_model_predict1 == 1") \
      .delegate_enrich(
        # 外流粗排UA模型预估接口
        kess_service = "{{explore_cascade_follow_ltv_model_predict_kess_service}}",
        request_type = "{{explore_cascade_follow_ltv_model_request_type}}",
        timeout_ms = 100,
        send_common_attrs =  [
          { "name": "userInfo", "as": "user_info_str" }
        ],
        recv_item_attrs = [
          { "name": "social_ltv_score1", "as": "social_ltv_score1" }
        ],
        for_predict = True,
        use_packed_item_attr = True,
        infer_output_type = 2
      ) \
    .end_()
