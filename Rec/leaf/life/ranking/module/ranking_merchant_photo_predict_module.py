from ranking import CommonModule

class RankingMerchantPhotoPredictModule(CommonModule):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def process(self) -> None:
        self.flow \
        .delegate_enrich(
          skip = "{{return enable_explore_rank_merchant_vedio_predict == 0}}",
          kess_service = "{{explore_rank_merchant_vedio_service}}",
          send_common_attrs = [
              { "name": "kuibaUserAttrStr", "as": "user_info_str" },
          ],
          recv_item_attrs = [
            {"name": "pcart_ctr", "as": "merchant_pcart_ctr"},
            {"name": "cvr", "as": "merchant_cart_cvr"},
            {"name": "gmv_fen", "as": "merchant_cart_gmv_fen"}
          ],
          timeout_ms = "{{explore_rank_merchant_vedio_predict_timeout_ms}}",
          request_type = "{{explore_rank_merchant_vedio_request_type}}",
          partition_size = "{{explore_rank_merchant_vedio_partition_size}}",
          target_item={"is_merchant_cart": 1},
          for_predict=True
        ) \
        .get_merchant_living_item_attr_by_distributed_index(
          skip = "{{return enable_explore_rank_merchant_living_predict == 0}}",
          photo_store_kconf_key = "reco.distributedIndex.exploreMerchantLivingPhotoStoreConfig",
          use_dynamic_photo_store = True,
          photo_store_rpc_req_cache_rate = 0,
          attrs = ["s_eshop_first_live_id"],
          item_id_attr = "merchant_author_in_living",
        ) \
        .delegate_enrich(
          skip = "{{return enable_explore_rank_merchant_living_predict == 0}}",
          kess_service = "{{explore_rank_merchant_living_service}}",
          send_common_attrs = [
            { "name": "kuibaUserAttrStr", "as": "user_info_str" },
          ],
          send_item_attrs = [
            {"name": "s_eshop_first_live_id", "as": "leaf_living_pId"}
          ],
          recv_item_attrs = [
            {"name": "elive_ctr", "as": "merchant_elive_ctr"},
            {"name": "elive_cvr", "as": "merchant_elive_cvr"},
            {"name": "elive_price", "as": "merchant_elive_price"}
          ],
          timeout_ms = "{{explore_rank_merchant_living_predict_timeout_ms}}",
          shard_num=1,
          request_type = "{{explore_rank_merchant_living_request_type}}",
          partition_size = "{{explore_rank_merchant_living_partition_size}}",
          for_predict=True,
          target_item={"is_merchant_living": 1}
        )