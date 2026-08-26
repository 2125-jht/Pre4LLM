from dragonfly.common_leaf_dsl import LeafFlow

class FountainCorrelationFlow(LeafFlow):
  def __init__(self, name: str):
    super().__init__(name)

    self \
      .enrich_attr_by_lua(
        import_common_attr = [
          "source_hetu_v2_face_id_list",
          "source_hetu_v2_tag_list",
          "source_hetu_v2_level_three_list",
          "source_hetu_v2_level_two_list",
          "source_hetu_v2_level_one_list",
        ],
        export_common_attr = [
          "source_hetu_v2_face_id_list",
          "source_hetu_v2_tag_list",
          "source_hetu_v2_level_three_list",
          "source_hetu_v2_level_two_list",
          "source_hetu_v2_level_one_list",
        ],
        function_for_common = "calculate",
        lua_script = """
          function calculate()
            local source_hetu_v2_face_id_list = source_hetu_v2_face_id_list or {}
            local new_source_hetu_v2_face_id_list = {}
            for i = 1, #source_hetu_v2_face_id_list do
              local tag_id = ((source_hetu_v2_face_id_list[i] >> 8) & 0xffffff)
              table.insert(new_source_hetu_v2_face_id_list, i, tag_id)
            end

            local source_hetu_v2_tag_list = source_hetu_v2_tag_list or {}
            local new_source_hetu_v2_tag_list = {}
            for i = 1, #source_hetu_v2_tag_list do
              local tag_id = ((source_hetu_v2_tag_list[i] >> 8) & 0xffffff)
              table.insert(new_source_hetu_v2_tag_list, i, tag_id)
            end

            local source_hetu_v2_level_three_list = source_hetu_v2_level_three_list or {}
            local new_source_hetu_v2_level_three_list = {}
            for i = 1, #source_hetu_v2_level_three_list do
              local tag_id = ((source_hetu_v2_level_three_list[i] >> 8) & 0xffffff)
              table.insert(new_source_hetu_v2_level_three_list, i, tag_id)
            end

            local source_hetu_v2_level_two_list = source_hetu_v2_level_two_list or {}
            local new_source_hetu_v2_level_two_list = {}
            for i = 1, #source_hetu_v2_level_two_list do
              local tag_id = ((source_hetu_v2_level_two_list[i] >> 8) & 0xffffff)
              table.insert(new_source_hetu_v2_level_two_list, i, tag_id)
            end

            local source_hetu_v2_level_one_list = source_hetu_v2_level_one_list or {}
            local new_source_hetu_v2_level_one_list = {}
            for i = 1, #source_hetu_v2_level_one_list do
              local tag_id = ((source_hetu_v2_level_one_list[i] >> 8) & 0xffffff)
              table.insert(new_source_hetu_v2_level_one_list, i, tag_id)
            end

            return new_source_hetu_v2_face_id_list, new_source_hetu_v2_tag_list,
              new_source_hetu_v2_level_three_list, new_source_hetu_v2_level_two_list,
              new_source_hetu_v2_level_one_list
          end
        """
      ) \
      .retrieve_by_remote_index(
        kess_service = "grpc_recoRelevanceTagOrderedIndexServer",
        timeout_ms = 100,
        reason = 10147,
        querys = [
          {
            "query": "hetu_tag_v2:{{source_hetu_v2_face_id_list}}",
          },
          {
            "query": "hetu_tag_v2:{{source_hetu_v2_tag_list}}",
          },
          {
            "query": "hetu_tag_v2:{{source_hetu_v2_level_three_list}}",
          },
          {
            "query": "hetu_tag_v2:{{source_hetu_v2_level_two_list}}",
          },
          {
            "query": "hetu_tag_v2:{{source_hetu_v2_level_one_list}}",
          },
        ],
        default_search_num = 20,
        default_expire_second = 300,
        default_total_request_num = 100,
      ) \
      .get_item_attr_by_distributed_flat_index(
        photo_store_kconf_key = "reco.distributedIndex.hotPhotoInfoCommonIndex",
        use_dynamic_photo_store = True,
        attrs = [
          "author__id",
          "hetu_tag_level_info__hetu_level_one",
          "hetu_tag_level_info__hetu_level_two",
          "author_age_info__age_segment",
          "fountain_stats__real_show_count",
          "fountain_stats__like_count",
          "fountain_stats__forward_count",
          "fountain_stats__follow_count",
          "fountain_stats__negative_count",
          "fountain_stats__view_length_sum",
        ],
      ) \
      .enrich_attr_by_lua(
        import_item_attr = [
          "fountain_stats__real_show_count",
          "fountain_stats__like_count",
          "fountain_stats__forward_count",
          "fountain_stats__follow_count",
          "fountain_stats__negative_count",
          "fountain_stats__view_length_sum",
        ],
        export_item_attr = [
          "empirical_ltr",
          "empirical_ftr",
          "empirical_wtr",
          "empirical_htr",
          "empirical_wtd",
        ],
        function_for_item = "calculate",
        lua_script = """
          function calculate(seq, item_key, reason, score)
            local total_count = fountain_stats__real_show_count or 0
            if total_count <= 0 then
              return 0.0, 0.0, 0.0, 0.0, 0.0
            end
            local like_count = fountain_stats__like_count or 0
            local forward_count = fountain_stats__forward_count or 0
            local follow_count = fountain_stats__follow_count or 0
            local negative_count = fountain_stats__negative_count or 0
            local view_length_sum = fountain_stats__view_length_sum or 0
            return like_count * 1.0 / total_count, forward_count * 1.0 / total_count,
              follow_count * 1.0 / total_count, negative_count * 1.0 / total_count,
              view_length_sum * 1.0 / total_count
          end
        """
      ) \
      .shuffle() \
      .limit(30) \
      .log_debug_info(
        common_attrs = [
          "page_size",
          "source_hetu_v2_face_id_list",
          "source_hetu_v2_tag_list",
          "source_hetu_v2_level_three_list",
          "source_hetu_v2_level_two_list",
          "source_hetu_v2_level_one_list",
        ],
        item_attrs = [
          "author__id",
          "hetu_tag_level_info__hetu_level_one",
          "hetu_tag_level_info__hetu_level_two",
          "author_age_info__age_segment",
          "fountain_stats__real_show_count",
          "fountain_stats__like_count",
          "fountain_stats__forward_count",
          "fountain_stats__follow_count",
          "fountain_stats__negative_count",
          "fountain_stats__view_length_sum",
        ],
        for_debug_request_only = True,
        respect_sample_logging = True,
      )
