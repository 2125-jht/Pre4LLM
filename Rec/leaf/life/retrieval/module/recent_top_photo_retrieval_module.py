from retrieval.retrieval_module import RetrievalModule

class RecentTopPhotoRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .if_("explore_recent_top_photo_retr_enable_bottom_select == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "_USER_ID_", "as": "user_id"},
            {"name": "_DEVICE_ID_", "as": "device_id"},
            {"name": "bottom_select_server_show_key_prefix", "as": "prefix"}
          ],
          export_common_attr = [
            {"name": "redis_key", "as": "bottom_select_server_show_redis_key"}
          ],
          function_name = "GenBottomSelectServerShowRedisKey",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .get_common_attr_from_redis(
          cluster_name = "recoDiversityOnline",
          redis_params = [
            {
              "redis_key": "{{bottom_select_server_show_redis_key}}",
              "output_attr_name": "bottom_select_server_show_redis_data_str"
            }
          ]
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "bottom_select_server_show_redis_data_str", "as": "redis_data_str"}
          ],
          export_common_attr = [
            {"name": "bottom_select_server_show_list", "as": "bottom_select_server_show_retrieval_list"}
          ],
          function_name = "ParseBottomSelectServerShowData",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("explore_recent_top_photo_retr_enable_u2u_select == 1") \
        .if_("explore_recent_top_photo_retr_enable_multi_target == 1") \
          .copy_attr(
            attrs=[{
              "from_common": "sim_user_list_multi_target",
              "to_common": "sim_user_list_total"
            }]
          ) \
        .end_() \
        .if_("explore_recent_top_photo_retr_enable_gcl_u2u == 1") \
          .pack_common_attr(
            input_common_attrs = [
              "sim_user_list_gcl_u2u",
              "sim_user_list_total"
            ],
            output_common_attr = "sim_user_list_total",
            deduplicate = True
          ) \
        .end_() \
        .if_("explore_recent_top_photo_retr_enable_mmu_u2u == 1") \
          .retrieve_by_ann_embedding(
            reason = 1,
            kess_service = "{{life_mmu_u2u_retr_service_name}}",
            space = "cosine",
            timeout_ms = 50,
            items_from_attr = ["_USER_ID_"],
            bound_type = {
              "top_k": "{{life_mmu_u2u_retr_user_num}}"
            },
            algo_type = {
              "scann": {}
            },
            src_data_type = "user",
            src_bucket = "user",
            dest_bucket = "{{life_mmu_u2u_retr_ann_dest_bucket}}",
            save_result_to_common_attr = "sim_user_list_mmu_u2u"
          ) \
          .pack_common_attr(
            input_common_attrs = [
              "sim_user_list_mmu_u2u",
              "sim_user_list_total"
            ],
            output_common_attr = "sim_user_list_total",
            deduplicate = True
          ) \
        .end_() \
        .if_("enable_life_retrieval_explore_recent_top_list == 1") \
          .retrieve_by_redis(
            reason = self.reason,
            retrieve_num = "{{u2u_cand_num}}",
            retrieve_num_per_key = "{{explore_recent_top_photo_retr_u2u_num_per_key}}",
            cluster_name = "recoExploreNegPhoto",
            timeout_ms = 20,
            key_from_attr = "sim_user_list_total",
            key_prefix = "{{explore_recent_top_photo_retr_u2u_key_prefix}}",
            item_separator = ",",
            save_result_to_common_attr = "explore_sim_user_recent_top_photo_retrieval_list"
          ) \
        .end_() \
        .if_("enable_life_retrieval_life_recent_top_list == 1") \
          .retrieve_by_redis(
            reason = self.reason,
            retrieve_num = "{{u2u_cand_num}}",
            retrieve_num_per_key = "{{explore_recent_top_photo_retr_u2u_num_per_key}}",
            cluster_name = "recoExploreNegPhoto",
            timeout_ms = 20,
            key_from_attr = "sim_user_list_total",
            key_prefix = "{{life_recent_top_photo_retr_u2u_key_prefix}}",
            item_separator = ",",
            save_result_to_common_attr = "life_sim_user_recent_top_photo_retrieval_list"
          ) \
        .end_() \
      .end_() \
      .pack_common_attr(
        input_common_attrs = [
          "bottom_select_server_show_retrieval_list",
          "explore_sim_user_recent_top_photo_retrieval_list",
          "life_sim_user_recent_top_photo_retrieval_list"
        ],
        output_common_attr = "recent_top_photo_id_list",
        deduplicate = True
      ) \
      .if_("explore_recent_top_photo_retr_enable_user_recent == 1") \
        .pack_common_attr(
          input_common_attrs = [
            "explore_rerank_pos_photo_id_retrieval_list",
            "recent_top_photo_id_list",
          ],
          output_common_attr = "recent_top_photo_id_list",
          deduplicate = True
        ) \
      .end_() \
      .if_("explore_recent_top_photo_retr_enable_user_rank_pos == 1") \
        .pack_common_attr(
          input_common_attrs = [
            "explore_rank_pos_photo_id_retrieval_list",
            "recent_top_photo_id_list",
          ],
          output_common_attr = "recent_top_photo_id_list",
          deduplicate = True
        ) \
      .end_() \
      .shuffle_list_attr(
        common_attr = "recent_top_photo_id_list"
      ) \
      .retrieve_by_common_attr(
        attr = "recent_top_photo_id_list",
        reason = self.reason
      ) \
      .filter_by_common_attr(
        common_attr = [
          "browse_screen__pid_list"
        ]
      ) \
      .filter_by_browse_set() \
      .limit(
        size = "{{cand_num}}"
      )
