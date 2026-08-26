from cascading import CommonModule

class CascadingMerchantPhotoPredictModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)


  def process(self) -> None:
    self.flow \
      .if_("enable_explore_mc_merchant_vedio_predict == 1") \
      .delegate_enrich( # 挂车短视频粗排PXTR预估
        kess_service="{{explore_mc_merchant_photo_double_tower_service}}",
        shard_num=1,
        partition_size="{{explore_mc_merchant_photo_double_tower_partition_size}}",
        use_packed_item_attr=True,
        send_common_attrs = [
          { "name": "kuibaUserAttrStr", "as": "user_info_str" },
        ],
        request_type="{{explore_mc_merchant_photo_double_tower_request_type}}",
        timeout_ms=100,
        infer_output_type=2,
        recv_item_attrs=[{
          "name": pred,
          "as": "merchant_tower_" + pred
        } for pred in ["ctr", "cvr", "gmv"]],
        target_item={"is_merchant_cart": 1},
        for_predict=True
      ) \
      .end_() \
      .if_("enable_explore_mc_merchant_living_predict == 1") \
      .delegate_enrich( # live头像短视频粗排PXTR预估
        kess_service="{{explore_mc_merchant_living_double_tower_service}}",
        shard_num=1,
        partition_size="{{explore_mc_merchant_living_double_tower_partition_size}}",
        use_packed_item_attr=True,
        send_common_attrs = [
          { "name": "kuibaUserAttrStr", "as": "user_info_str" },
        ],
        request_type="{{explore_mc_merchant_living_double_tower_request_type}}",
        timeout_ms=100,
        infer_output_type=2,
        recv_item_attrs=[{
          "name": pred,
          "as": "merchant_elive_tower_" + pred
        } for pred in ["ctr", "cvr", "gmv"]],
        target_item={"is_merchant_living": 1},
        for_predict=True
      ) \
      .end_()
