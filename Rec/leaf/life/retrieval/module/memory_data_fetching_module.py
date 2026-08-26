from retrieval import CommonModule

class MemoryDataFetchingModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self):
    self.flow \
      .explore_memory_data_enrich(
        data_key = "explore_mc_dynamic_xtr_debias_map",
        data_type = "string_double_vector_map",
        save_data_ptr_to_attr = "explore_mc_hourly_xtr_debias_map_ptr",
      ) \
      .explore_memory_data_enrich(
        data_key = "explore_fullrank_dynamic_xtr_debias_map",
        data_type = "string_double_vector_map",
        save_data_ptr_to_attr = "explore_fullrank_hourly_xtr_debias_map_ptr",
      ) \
      .explore_memory_data_enrich(
        data_key="explore_support_author_tgi_score",
        data_type="uint64_uint64_double_map_map",
        save_data_ptr_to_attr="support_author_memory_data"
      ) \
      .explore_memory_data_enrich(
        data_key="prerank_duration_debias_bucket",
        data_type="string_double_vector_map",
        save_data_ptr_to_attr="prerank_duration_debias_bucket"
      ) \
      .explore_memory_data_enrich(
        data_key = "xhs_hetu_tags",
        data_type = "uint64_set",
        save_data_ptr_to_attr = "xhs_hetu_memorydata_set",
      ) \
      .explore_memory_data_enrich(
        data_key = "hetu_v1_id_mapping",
        data_type = "uint64_uint64_map",
        save_data_ptr_to_attr = "hetu_v1_id_mapping_ptr"
      ) \
      .explore_memory_data_enrich(
        data_key = "merchant_live_authors_set",
        data_type = "uint64_set",
        save_data_ptr_to_attr = "merchant_live_authors_set__memory_data",
      ) \
      .explore_memory_data_enrich(
        data_key = "explore_hetu_gender_debias",
        data_type = "string_double_vector_map",
        save_data_ptr_to_attr = "hetu_gender_to_norm_ctr_map_ptr"
      ) \
      .explore_memory_data_enrich(
        data_key = "explore_personifed_author_pctr_ctr",
        data_type = "string_double_map",
        save_data_ptr_to_attr = "explore_personifed_author_boost_ptr"
      ) \
      .explore_memory_data_enrich(
        data_key = "explore_pic_xtr_fractile_score_map",
        data_type = "string_double_vector_map",
        save_data_ptr_to_attr = "pic_xtr_fractile_score_attr_from_redis_ptr",
      ) \
      .explore_memory_data_enrich(
        data_key = "explore_pic_xtr_cluster_emp_map",
        data_type = "string_double_vector_map",
        save_data_ptr_to_attr = "explore_pic_xtr_cluster_emp_map_ptr",
      ) \
      .explore_memory_data_enrich(
        data_key = "explore_pic_fr_pxtr_pcts",
        data_type = "string_double_map",
        save_data_ptr_to_attr = "explore_pic_fr_pxtr_pcts_ptr",
      ) \
      .explore_memory_data_enrich(
        data_key = "explore_pic_da_user_pref_score_data",
        data_type = "string_double_map",
        save_data_ptr_to_attr = "pic_da_user_pref_ptr",
      ) \
      .explore_memory_data_enrich(
        data_key = "explore_pic_fr_rel_score_pct_map",
        data_type = "string_double_vector_map",
        save_data_ptr_to_attr = "pic_fr_rel_score_pct_map",
      ) \
      .explore_memory_data_enrich(
        data_key = "explore_pic_rerank_pid_realshow_data_map",
        data_type = "string_double_vector_map",
        save_data_ptr_to_attr = "pic_rerank_pid_realshow_data_map",
      ) \
      .explore_memory_data_enrich(
        data_key = "high_value_black_author_map",
        data_type = "uint64_double_vector_map",
        save_data_ptr_to_attr = "high_value_black_author_map_ptr",
      ) \
      .explore_memory_data_enrich(
        data_key = "negative_aid",
        data_type = "uint64_set",
        save_data_ptr_to_attr = "negative_aid_set_ptr",
      ) \
      .if_("explore_enable_fetch_rank_neg_photo == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_rank_neg_photo_key_prefix", "as": "string_a"},
            {"name": "_DEVICE_ID_", "as": "string_b"}
          ],
          export_common_attr = [
            {"name": "final_string", "as": "explore_rank_neg_photo_redis_key"}
          ],
          function_name = "ConcatString",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .get_common_attr_from_redis( #上一刷精排结果过滤
          cluster_name = "recoExploreNegPhoto",
          redis_params = [
            {
              "redis_key": "{{explore_rank_neg_photo_redis_key}}",
              "output_attr_name": "rank_neg_photo_id_list_str"
            }
          ]
        ) \
      .end_() \
      .if_("explore_enable_fetch_rerank_neg_photo == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_rerank_neg_photo_key_prefix", "as": "string_a"},
            {"name": "_DEVICE_ID_", "as": "string_b"}
          ],
          export_common_attr = [
            {"name": "final_string", "as": "explore_rerank_neg_photo_redis_key"}
          ],
          function_name = "ConcatString",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .get_common_attr_from_redis( #上一刷精排结果过滤
          cluster_name = "recoExploreNegPhoto",
          redis_params = [
            {
              "redis_key": "{{explore_rerank_neg_photo_redis_key}}",
              "output_attr_name": "rerank_neg_photo_id_list_str"
            }
          ]
        ) \
      .end_() \
      .if_("explore_enable_fetch_explore_rerank_pos_photo == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_rerank_pos_photo_key_prefix", "as": "string_a"},
            {"name": "_DEVICE_ID_", "as": "string_b"}
          ],
          export_common_attr = [
            {"name": "final_string", "as": "explore_rerank_pos_photo_redis_key"}
          ],
          function_name = "ConcatString",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .get_common_attr_from_redis(
          cluster_name = "recoExploreNegPhoto",
          redis_params = [
            {
              "redis_key": "{{explore_rerank_pos_photo_redis_key}}",
              "output_attr_name": "explore_rerank_pos_photo_id_list_str"
            }
          ]
        ) \
        .split_string(
          input_common_attr = "explore_rerank_pos_photo_id_list_str",
          output_common_attr = "explore_rerank_pos_photo_id_retrieval_list",
          delimiters = ",",
          trim_spaces = True,
          skip_empty_tokens = True,
          parse_to_int = True
        ) \
        .filter_by_browse_set(
          item_list_from_attr = "explore_rerank_pos_photo_id_retrieval_list"
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_rerank_pos_photo_id_retrieval_list", "as": "universal_set_list"},
            {"name": "browse_screen__pid_list", "as": "sub_set_list"}
          ],
          export_common_attr = [
            {"name": "difference_list", "as": "explore_rerank_pos_photo_id_retrieval_list"}
          ],
          function_name = "GetDifferenceSet",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("explore_enable_fetch_cascade_neg_photo == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_cascade_neg_photo_key_prefix", "as": "string_a"},
            {"name": "_DEVICE_ID_", "as": "string_b"}
          ],
          export_common_attr = [
            {"name": "final_string", "as": "explore_cascade_neg_photo_redis_key"}
          ],
          function_name = "ConcatString",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .get_common_attr_from_redis(
          cluster_name = "recoExploreUserStat",
          redis_params = [
            {
              "redis_key": "{{explore_cascade_neg_photo_redis_key}}",
              "output_attr_name": "cascade_neg_photo_id_list_str"
            }
          ]
        ) \
        .split_string(
          input_common_attr = "cascade_neg_photo_id_list_str",
          output_common_attr = "cascade_neg_photo_id_list",
          delimiters = ",",
          parse_to_int = True,
          trim_spaces = True,
          skip_empty_tokens = True
        ) \
      .end_() \
      .if_("life_enable_week_cluster_get_redis == 1") \
        .str_format(
          format_string= "l_vc%d",
          input_attrs=["_USER_ID_"],
          output_attr="week_valid_cluster_redis_key",
        ) \
        .get_common_attr_from_redis(
          name="get_common_attr_from_redis_week_cluster_str",
          cluster_name = "recoUserGroup",
          redis_params = [
            {
              "redis_key": "{{week_valid_cluster_redis_key}}",
              "redis_value_type": "string",
              "output_attr_name": "week_cluster_str",
              "output_attr_type": "string"
            }
          ],
          timeout_ms = 5,
          is_async = True
        )\
        .split_string(
          input_common_attr = "week_cluster_str",
          output_common_attr = "week_cluster_list", 
          delimiters=",",
          parse_to_int=True,
        )\
      .end_() \
      .if_("life_enable_intere_cluster_get_redis == 1") \
        .str_format(
          format_string= "y_c_id%d",
          input_attrs=["_USER_ID_"],
          output_attr="intere_cluster_redis_key",
        ) \
        .get_common_attr_from_redis(
          name="get_common_attr_from_redis_intere_cluster_str",
          cluster_name = "kkdKsUidClickList",
          redis_params = [
            {
              "redis_key": "{{intere_cluster_redis_key}}",
              "redis_value_type": "string",
              "output_attr_name": "cluster_id_str",
              "output_attr_type": "string"
            }
          ],
          timeout_ms = 5,
          is_async = True
        )\
        .split_string(
          input_common_attr = "cluster_id_str",
          output_common_attr = "cluster_id_list", 
          delimiters=",",
          parse_to_int=True,
        )\
      .end_() \
      .if_("life_enable_intere_score_get_redis == 1") \
        .str_format(
          format_string= "y_c_in%d",
          input_attrs=["_USER_ID_"],
          output_attr="intere_score_redis_key",
        ) \
        .get_common_attr_from_redis(
          name="get_common_attr_from_redis_score_cluster_str",
          cluster_name = "kkdKsUidClickList",
          redis_params = [
            {
              "redis_key": "{{intere_score_redis_key}}",
              "redis_value_type": "string",
              "output_attr_name": "intere_score_str",
              "output_attr_type": "string"
            }
          ],
          timeout_ms = 5,
          is_async = True
        )\
        .split_string(
          input_common_attr = "intere_score_str",
          output_common_attr = "intere_score_list", 
          delimiters=",",
          parse_to_double=True,
        )\
      .end_() \
      .if_("life_enable_intere_score_get_redis_v2 == 1") \
        .str_format(
          format_string= "combo_y_c%d",
          input_attrs=["_USER_ID_"],
          output_attr="intere_score_redis_key",
        ) \
        .get_common_attr_from_redis(
          name="get_common_attr_from_redis_combo_cluster_str",
          cluster_name = "kkdKsUidClickList",
          redis_params = [
            {
              "redis_key": "{{intere_score_redis_key}}",
              "redis_value_type": "string",
              "output_attr_name": "combo_intere_str",
              "output_attr_type": "string"
            }
          ],
          timeout_ms = 5,
          is_async = True
        )\
        .split_string(
          input_common_attr = "combo_intere_str",
          output_common_attr = "intere_str_list", 
          delimiters=":",
        )\
      .end_() \
      

  def post_process(self) -> None:
    self.flow \
      .log_debug_info(
        # item_attr = ["hetu_tag_level_info__hetu_cluster_id" ],
        common_attrs = ["explore_fullrank_hourly_xtr_debias_map_ptr", "hetu_gender_to_norm_ctr_map_ptr", "explore_rerank_pos_photo_id_retrieval_list",
                        "pic_da_user_pref_ptr", "pic_fr_rel_score_pct_map", "pic_rerank_pid_realshow_data_map", "week_cluster_list",
                        "cluster_id_list", "intere_score_list","intere_cluster_redis_key","cluster_id_str","intere_score_str",
                        "combo_intere_str","intere_str_list", ],
        for_debug_request_only = True,
      )