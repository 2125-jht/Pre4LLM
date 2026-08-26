from retrieval.retrieval_module import RetrievalModule

class RecentTopPhotoRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .if_("fountain_recent_top_photo_retr_enable_bottom_select == 1", to_be_delete = "date=2024-05-29;committer=liuhao07") \
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
      .if_("fountain_recent_top_photo_retr_enable_u2u_select_fast == 1") \
        .if_("fountain_recent_top_photo_retr_enable_multi_target_fast == 1") \
          .copy_attr(
            attrs=[{
              "from_common": "sim_user_list_multi_target",
              "to_common": "sim_user_list_total"
            }]
          ) \
        .end_() \
        .if_("fountain_recent_top_photo_retr_enable_gcl_u2u_fast == 1") \
          .pack_common_attr(
            input_common_attrs = [
              "gcl_sim_user_list",
              "sim_user_list_total"
            ],
            output_common_attr = "sim_user_list_total",
            deduplicate = True
          ) \
        .end_() \
        .if_("fountain_recent_top_photo_retr_enable_mc_u2u_fast == 1") \
          .pack_common_attr(
            input_common_attrs = [
              "focal_user_list",
              "sim_user_list_total"
            ],
            output_common_attr = "sim_user_list_total",
            deduplicate = True
          ) \
        .end_() \
        .retrieve_by_redis(
          reason = 500,
          retrieve_num = "{{u2u_cand_num}}",
          retrieve_num_per_key = "{{fountain_recent_top_photo_retr_u2u_num_per_key_fast}}",
          cluster_name = "recoExploreNegPhoto",
          timeout_ms = 10,
          key_from_attr = "sim_user_list_total",
          key_prefix = "{{fountain_recent_top_photo_retr_u2u_key_prefix}}",
          item_separator = ",",
          save_result_to_common_attr = "fountain_sim_user_recent_top_photo_retrieval_list"
        ) \
      .end_() \
      .pack_common_attr(
        input_common_attrs = [
          "explore_rank_pos_photo_id_retrieval_list",
          "bottom_select_server_show_retrieval_list",
          "fountain_sim_user_recent_top_photo_retrieval_list"
        ],
        output_common_attr = "recent_top_photo_id_list",
        deduplicate = True
      ) \
      .if_("fountain_recent_top_photo_retr_enable_user_recent_fast == 1", to_be_delete = "date=2024-05-29;committer=liuhao07") \
        .pack_common_attr(
          input_common_attrs = [
            "fountain_rerank_pos_photo_id_retrieval_list",
            "recent_top_photo_id_list",
          ],
          output_common_attr = "recent_top_photo_id_list",
          deduplicate = True
        ) \
      .end_() \
      .if_("fountain_recent_top_photo_retr_enable_user_rank_pos_fast == 1", to_be_delete = "date=2024-05-29;committer=liuhao07") \
        .pack_common_attr(
          input_common_attrs = [
            "fountain_rank_pos_photo_id_retrieval_list",
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
      .limit(
        size = "{{cand_num}}"
      )