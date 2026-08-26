from cascading import CommonModule

class InterestMigrationDataPrepareModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self):
    self.flow \
      .if_(" life_enable_single_interest_migration_data == 1") \
        .split_string(
          input_common_attr = "interest_migration_ignore_channel_list_str",
          output_common_attr = "interest_migration_ignore_channel_list",
          delimiters = ",", 
          skip_empty_tokens = True,
          trim_spaces = True,
          parse_to_int = True 
        ) \
        .explore_interest_migration_history_prepare_enricher(
          time_second_upper = "{{interest_migration_time_second_upper}}",
          colossus_v2_attr_name = "colossus_resp_v2",
          user_info_ptr_name = "user_info_ptr",
          ignore_channel_name = "interest_migration_ignore_channel_list",
          output_id_list_name = "interest_migration_pids",
          output_score_list_name = "interest_migration_scores",
          output_realshow_list_name = "explore_realshow_pids",
          output_is_degraded_flag_name = "interest_migration_is_degraded",
          colossus_num_limit_attr = "{{interest_migration_colossus_num_limit}}",
          realshow_num_limit_attr = "{{interest_migration_realshow_num_limit}}",
          hot_cnt_threshold_attr = "{{interest_migration_hot_cnt_threshold}}",
          hot_rate_threshold_attr = "{{interest_migration_hot_rate_threshold}}",
          playtime_weight = "{{interest_migration_playtime_weight}}",
          not_effective_view_weight = "{{interest_migration_not_effective_view_weight}}",
          like_weight_attr = "{{interest_migration_like_weight}}",
          follow_weight_attr = "{{interest_migration_follow_weight}}",
          forward_weight_attr = "{{interest_migration_forward_weight}}",
          comment_weight_attr = "{{interest_migration_comment_weight}}",
          profile_weight_attr = "{{interest_migration_profile_weight}}",
          collection_weight_attr = "{{interest_migration_collection_weight}}",
          short_rate_hour_upper_attr = "{{interest_migration_short_hour_upper}}",
          short_rate_cnt_threshold_attr = "{{interest_migration_short_rate_cnt_threshold}}",
          long_rate_vv_threshold_attr = "{{interest_migration_long_rate_vv_threshold}}",
          vv_rate_weight_attr = "{{interest_migration_vv_rate_weight}}",
          play_time_rate_weight_attr = "{{interest_migration_play_time_rate_weight_attr}}",
          active_rate_weight_attr = "{{interest_migration_active_rate_weight_attr}}",
          output_user_page_prefer_score_name = "user_page_prefer_score",
          enable_collection_list_attr = "{{interest_migration_enable_collection_list}}",
          bs_short_view_output_id_list_name = "bs_short_view_pids",
          bs_short_view_mins_upper = "{{interest_migration_bs_short_view_mins_upper}}",
          bs_short_view_num_upper = "{{interest_migration_bs_short_view_num_upper}}",
        ) \
        .split_string(
          input_common_attr = "interest_migration_ignore_cluster_lv1_classes_str",
          output_common_attr = "interest_migration_ignore_cluster_lv1_classes",
          delimiters = ",", 
          skip_empty_tokens = True,
          trim_spaces = True,
          parse_to_int = True 
        ) \
        .enrich_attr_by_lua(
          import_common_attr = [
            "colossus_photo_id_list",
            "colossus_channel_list",
          ],
          export_common_attr = [
            "explore_realshow_pids",
          ],
          function_for_common = "calculate",
          lua_script = """
            function calculate()
              local life_realshow_pids = {}
              if colossus_photo_id_list~=nil and colossus_channel_list ~= nil and 
                #colossus_channel_list == #colossus_photo_id_list and #colossus_photo_id_list >0 then
                for i=1 , #colossus_photo_id_list do
                  local channel = colossus_channel_list[i]
                  local pid = colossus_photo_id_list[i]
                  if channel == 125 or channel == 122 then
                    table.insert(life_realshow_pids, colossus_photo_id_list[i])
                  end
                end
              end
              return life_realshow_pids
            end
          """,
        ) \
        .get_item_attr_by_distributed_flat_index(
          photo_store_kconf_key = "reco.distributedIndex.hotPhotoInfoCommonIndex",
          use_dynamic_photo_store = True,
          photo_store_request_data_set_tags_attr = 'explore_request_data_set_tags',
          attrs = ["hetu_sim_cluster_id",],
          additional_item_source={
            "reco_results": True,
            "common_attr": [ "interest_migration_pids", "explore_realshow_pids"]
          }
        ) \
        .get_kconf_params(
          kconf_configs = [{
            "kconf_key": "reco.offline.cidGroupMapStr2Int",
            "json_path": "{{hetu_sim_cluster_id}}",
            "default_value": -1,
            "export_item_attr": "hetu_sim_cluster_id862_lv1",
          }]
        ) \
        .explore_interest_migration_coef_calculator_enricher( 
          explore_realshow_pids_attr = "explore_realshow_pids",
          gamora_play_pids_attr = "interest_migration_pids",
          gamora_play_scores_attr = "interest_migration_scores",
          cluster_id_attr = "hetu_sim_cluster_id",
          output_coef_attr = "interest_migration_photo_coef",
          gamora_score_threshold = "{{interest_migration_score_threshold}}",
          migration_threshold = "{{interest_migration_migration_threshold}}",
          migration_coef = "{{interest_migration_migration_coef}}",
          cluster_id_lv1_attr = "hetu_sim_cluster_id862_lv1",
          filter_by_cluster_lv1_classes_attr = "interest_migration_ignore_cluster_lv1_classes",
          output_hot_rate_attr = "cid_hot_show_rate",
        )\
        .set_attr_value(
          no_overwrite=True,
          item_attrs=[
            {
              "name": "interest_migration_operation_coefficient",
              "type": "double",
              "value": 0.1
            }
          ]
        )\
        .item_attr_operation(
          item_attr_a="interest_migration_photo_coef",
          item_attr_b="interest_migration_operation_coefficient",
          operator="*",
          output_attr="interest_migration_photo_coef"
        )\
        .pack_item_attr(
          item_source = {
            "reco_results": True,
          },
          mappings = [
            {
              "aggregator": "concat",
              "from_item_attr": "interest_migration_photo_coef",
              "to_common_attr": "interest_migration_photo_coef_list",
              "default_val": -1,
            },
            # {
            #   "aggregator": "concat",
            #   "from_item_attr": "interest_migration_photo_coef2",
            #   "to_common_attr": "interest_migration_photo_coef2_list",
            #   "default_val": -1,
            # },
            # {
            #   "aggregator": "concat",
            #   "from_item_attr": "cid_hot_show_rate",
            #   "to_common_attr": "cid_hot_show_rate_list",
            #   "default_val": -1,
            # },
            # {
            #   "aggregator": "concat",
            #   "from_item_attr": "hetu_sim_cluster_id862_lv1",
            #   "to_common_attr": "hetu_sim_cluster_id862_lv1_list",
            #   "default_val": -1,
            # },
            # {
            #   "aggregator": "concat",
            #   "from_item_attr": "norm_interest_migration_photo_coef",
            #   "to_common_attr": "norm_interest_migration_photo_coef_list",
            #   "default_val": -1,
            # },
          ],
        ) \
        .log_debug_info(
          common_attrs = [
            "interest_migration_photo_coef_list",
            # "norm_interest_migration_photo_coef_list",
            # "interest_migration_photo_coef2_list",
            # "cid_hot_show_rate_list", "hetu_sim_cluster_id862_lv1_list",
            "interest_migration_scores",
            "explore_realshow_pids","interest_migration_is_degraded","bs_short_view_pids",
            "user_page_prefer_score",
          ],
          item_attrs = [ "hetu_sim_cluster_id","interest_migration_photo_coef","cid_hot_show_rate","hetu_sim_cluster_id862_lv1",],
          for_debug_request_only = True,
          respect_sample_loggging = True,
        )\
      .end_() \
      
