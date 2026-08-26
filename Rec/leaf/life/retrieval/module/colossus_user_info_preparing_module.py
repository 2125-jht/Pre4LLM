from retrieval import CommonModule

class ColossusUserInfoPreparingModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self):
    self.flow \
      .if_("enable_use_colossus_old == 1") \
        .colossus(
          service_name = "grpc_colossusSim",
          client_type = "sim_client",
          output_attr = "colossus_resp_old"
        ) \
      .end_() \
      .colossus(
        service_name = "grpc_colossusSimV2",
        client_type = "common_item_client",
        output_attr = "colossus_resp_v2",
        parse_to_pb = False,
      ) \
      .gsu_common_colossus_resp_retriever(
        colossus_resp_attr = "colossus_resp_v2",
        colossus_service_name = "grpc_colossusSimV2",
        item_key_field = "photo_id",
        item_time_field = "timestamp",
        item_fields = dict(
          photo_id = "colossus_photo_id_list",
          play_time = "colossus_play_time_list",
          label = "colossus_label_list",
          author_id = "colossus_author_id_list",
          channel = "colossus_channel_list",
          duration = "colossus_duration_list",
          timestamp = "colossus_timestamp_list",
          tag = "colossus_tag_list"
        ),
        to_common_attr = True,
        max_item_num = 800,
      ) \
      .copy_attr(
        attrs=[{
          "from_common": "colossus_photo_id_list",
          "to_common": "copy_colossus_photo_id_list"
        }]
      ) \
      .get_item_attr_by_distributed_flat_index(
        photo_store_kconf_key = "reco.distributedIndex.hotPhotoInfoCommonIndex",
        use_dynamic_photo_store = True,
        photo_store_request_data_set_tags_attr = 'explore_request_data_set_tags',
        attrs = [
          "hetu_tag_level_info__hetu_level_one",
          ],
        additional_item_source={
          "reco_results": False,
          "common_attr": ["copy_colossus_photo_id_list"],
        },
      ) \
      .pack_item_attr(
        item_source = {
          "reco_results": False,
          "common_attr": ["copy_colossus_photo_id_list"],
        },
        mappings = [
        {
          "from_item_attr": "hetu_tag_level_info__hetu_level_one",
          "to_common_attr": "colossus_hetu_one_list",
          "default_val" : -1,
        },
        ]
      )\
      .if_("enable_use_colossus_resp_v2 == 1") \
      .get_kconf_params(
        kconf_configs = [{
          "kconf_key": "reco.interestExplore.remapClusterId632",
          "value_type": "list_int64",
          "export_common_attr": "remap_cluster_id_632_list",
          "default_value": []
        }]
      ) \
      .get_item_attr_by_distributed_flat_index(
          photo_store_kconf_key = "reco.distributedIndex.hotPhotoInfoCommonIndex",
          use_dynamic_photo_store = True,
          photo_store_request_data_set_tags_attr = 'explore_request_data_set_tags',
          attrs = ["hetu_sim_cluster_id"],
          additional_item_source={
            "reco_results": False,
            "common_attr": ["copy_colossus_photo_id_list"],
          },
        ) \
      .pack_item_attr(
        item_source = {
          "reco_results": False,
          "common_attr": ["copy_colossus_photo_id_list"],
        },
        mappings = [
        {
          "from_item_attr": "hetu_sim_cluster_id",
          "to_common_attr": "colossus_cluster_id_list",
          "default_val" : -1,
        },
        ]
      )\
      .end_() \
      .log_debug_info(
        common_attrs = [
          "colossus_hetu_one_list",
           "colossus_cluster_id_list",
          "remap_cluster_id_632_list",
          "colossus_photo_id_list",
          "colossus_play_time_list",
          "colossus_label_list",
          "colossus_author_id_list",
          "colossus_channel_list",
          "colossus_duration_list",
          "colossus_timestamp_list",
          "colossus_tag_list"
        ],
        for_debug_request_only = True,
        respect_sample_loggging = True,
      ) \
      .if_("enable_life_use_colossus_new == 1") \
        .gsu_common_colossusv2_enricher(
          kconf='colossus.kconf_client.video_item',
          limit="{{life_colossus_new_limit_num}}",
          partial_schema_output_attr="video_item_schema",
          item_fields=dict(
            photo_id="colossus_photo_id_list_new",     
            # 使用修复后的 author_id_v2
            author_id_v2="colossus_author_id_list_new",
            duration="colossus_duration_list_new",
            play_time="colossus_play_time_list_new",
            tag="colossus_tag_list_new",
            channel="colossus_channel_list_new",
            label="colossus_label_list_new",
            timestamp="colossus_timestamp_list_new")) \
        .gsu_common_colossus_resp_retriever(
          from_colossus_sim_v2=True,
          # debug 专用，通常不要配
          # print_item_fields=True,
          partial_schema_input_attr="video_item_schema",
          item_key_field="photo_id",
          item_time_field="timestamp",
          input_item_fields=dict(
            photo_id="colossus_photo_id_list_new",
            # 这里必须是 author_id_v2
            author_id_v2="colossus_author_id_list_new",
            duration="colossus_duration_list_new",
            play_time="colossus_play_time_list_new",
            tag="colossus_tag_list_new",
            channel="colossus_channel_list_new",
            label="colossus_label_list_new",
            timestamp="colossus_timestamp_list_new"),
          item_fields=dict(photo_id="pids",
            # 这里必须是 author_id_v2
            author_id_v2="aids",
            duration="drs",
            play_time="pls",
            tag="tgs",
            channel="cns",
            label="lbs",
            timestamp="tss")) \
        .log_debug_info(
          common_attrs = [
            "colossus_photo_id_list_new",
            "colossus_author_id_list_new",
            "colossus_duration_list_new",
            "colossus_play_time_list_new",
            "colossus_tag_list_new",
            "colossus_channel_list_new",
            "colossus_label_list_new",
            "colossus_timestamp_list_new",
            "life_colossus_new_limit_num",
          ],
          item_attrs = [
            "pids",
            "aids",
            "drs",
            "pls",
            "tgs",
            "cns",
            "lbs",
            "tss",
          ],
          for_debug_request_only = True,
          respect_sample_loggging = True,
        ) \
      .end_() \
      .if_("enable_life_realtime200_action_prepare == 1") \
        .enrich_attr_by_lua(
          import_common_attr = ["colossus_photo_id_list_new", "colossus_play_time_list_new"],
          export_common_attr = ["colossus_photo_id_list_new_positive"],
          function_for_common = "calculate_longplay_trigger",
          lua_script = """
              function calculate_longplay_trigger()
                  local colossus_photo_id_list_new = colossus_photo_id_list_new or {}
                  local colossus_play_time_list_new = colossus_play_time_list_new or {}
                  local colossus_photo_id_list_new_positive = {}
                  
                  if #colossus_photo_id_list_new == #colossus_play_time_list_new then
                    for i = #colossus_photo_id_list_new, 1, -1 do
                      if #colossus_photo_id_list_new_positive < 200 and colossus_play_time_list_new[i] >= 7 then
                        table.insert(colossus_photo_id_list_new_positive, colossus_photo_id_list_new[i])
                      end
                    end
                  end
              return colossus_photo_id_list_new_positive
          end
          """
        ) \
        .log_debug_info(
          common_attrs = ["colossus_photo_id_list_new", "colossus_play_time_list_new", "colossus_photo_id_list_new_positive"],
          for_debug_request_only = True,
          respect_sample_loggging = True,
        ) \
      .end_() \
      .if_("enable_life_realtime20_action_prepare == 1") \
        .enrich_attr_by_lua(
            import_common_attr = [
              "colossus_photo_id_list_new", "colossus_play_time_list_new", "colossus_timestamp_list_new", "browse_screen__pid_list",
              "enable_life_realtime_action_sensitive_browse_nums"
            ], 
            export_common_attr = ["realtime_photo_id_list", "is_realtime_boost"],
            function_for_common = "calculate_realtime_list",
            lua_script_file = "life/retrieval/lua/module/colossus_ann__gen_trigger_weight.lua"
          ) \
        .log_debug_info(
          common_attrs = ["realtime_photo_id_list", "is_realtime_boost", "colossus_photo_id_list_new", "browse_screen__pid_list"],
          for_debug_request_only = True,
          respect_sample_loggging = True,
        ) \
      .end_() \
      .if_("negative_trigger_num > 0") \
        .fetch_kgnn_neighbors(
          id_from_common_attr = "_USER_ID_",
          save_neighbors_to = "negative_trigger_ids",
          save_weight_to = "negative_trigger_weights",
          kess_service = "grpc_kgnn_explore_user_trigger_graph-U2I",
          relation_name = "U2I",
          shard_num = 1,
          sample_num = "{{negative_trigger_num}}",
          timeout_ms = 50,
          sample_type = "topn",
          padding_type = "zero"
        ) \
      .end_() \
      .if_("prefer_trigger_num > 0") \
        .fetch_kgnn_neighbors(
          id_from_common_attr = "_USER_ID_",
          save_neighbors_to = "prefer_trigger_ids",
          save_weight_to = "prefer_trigger_weights",
          kess_service = "grpc_kgnn_explore_user2trigger_graph-U2I",
          relation_name = "U2I",
          shard_num = 1,
          sample_num = "{{prefer_trigger_num}}",
          timeout_ms = 50,
          sample_type = "topn",
          padding_type = "no_padding"
        ) \
      .end_()
        