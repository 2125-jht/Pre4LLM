from cascading import CommonModule

class CascadingUninterestDepressScoreModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
      self.flow \
        .if_("enable_life_realtime_uninterest_deboost_score == 1") \
         .get_kconf_params(
            kconf_configs = [{
            "kconf_key": "reco.interestExplore.remapClusterId632",
            "export_common_attr": "remapClusterList",
            "value_type": "list_int64",
            }]
          ) \
          .get_item_attr_by_distributed_flat_index(
            photo_store_kconf_key = "reco.distributedIndex.hotPhotoInfoCommonIndex",
            use_dynamic_photo_store = True,
            photo_store_request_data_set_tags_attr = 'explore_request_data_set_tags',
            attrs = [
              "hetu_sim_cluster_id", 
              ],
            additional_item_source={
              "reco_results": False,
              "common_attr": ["browse_screen__pid_list"],
            },
          ) \
          .pack_item_attr(
              item_source={
                "reco_results": False,
                "common_attr": ["browse_screen__pid_list"],
                },
              mappings=[
                {
                  "aggregator": "concat",
                  "from_item_attr": "hetu_sim_cluster_id",
                  "to_common_attr": "concat_browset_cluster1000_list",
                  "default_val": 0,
                }
              ]
          ) \
          .enrich_attr_by_lua(
            import_common_attr = [
              "colossus_photo_id_list_new_positive", "browse_screen__pid_list", "concat_browset_cluster1000_list", "remapClusterList", "enable_life_realtime_uninterest_num_thred",
            ],  
            export_common_attr = ["uninterest_cluster_list", "interest_test_k", "interest_test_v", "uninterest_test_k", "uninterest_test_v"],
            function_for_common = "calculate_uninterest_cluster",
            lua_script="""
            function calculate_uninterest_cluster() 
                local num_thred = enable_life_realtime_uninterest_num_thred or 5             
                local click_list = colossus_photo_id_list_new_positive or {}
                local browset = browse_screen__pid_list or {}
                local browset_cluster = concat_browset_cluster1000_list or {}
                local interest_map = {}
                local uninterest_map = {}
                if #browset == #browset_cluster and remapClusterList ~= nil then
                  for i = #browset, math.max(1, #browset - 200), -1 do
                    local mapping_cluster = remapClusterList[(browset_cluster[i] or 0) + 1]
                    for j = 1, #click_list do  
                      if browset[i] == click_list[j] then
                        interest_map[mapping_cluster] = (interest_map[mapping_cluster] or 0) + 1
                        break
                      end
                    end
                    uninterest_map[mapping_cluster] = (uninterest_map[mapping_cluster] or 0) + 1
                  end
                end
                local uninterest_cluster_list = {}
                for key, value in pairs(uninterest_map or {}) do
                  if (uninterest_map[key] or 0) >= num_thred and (interest_map[key] or 0) == 0 then
                    table.insert(uninterest_cluster_list, key)
                  end
                end

                local interest_test_k = {}
                local interest_test_v = {}
                local uninterest_test_k = {}
                local uninterest_test_v = {}
                for key, value in pairs(interest_map or {}) do
                  table.insert(interest_test_k, key)
                  table.insert(interest_test_v, value)
                end
                for key, value in pairs(uninterest_map or {}) do
                  table.insert(uninterest_test_k, key)
                  table.insert(uninterest_test_v, value)
                end

                return uninterest_cluster_list, interest_test_k, interest_test_v, uninterest_test_k, uninterest_test_v
            end
            """
          ) \
          .enrich_attr_by_lua(
            import_common_attr = ["uninterest_cluster_list", "remapClusterList"],  
            import_item_attr = ["hetu_sim_cluster_id"],  
            export_item_attr = ["is_uninterest_depress", "item_cluster632"],
            function_for_item = "calculate_uninterest_cluster",
            lua_script="""
            function calculate_uninterest_cluster()              
                local uninterest_cluster_list = uninterest_cluster_list or {}
                local item_cluster632 = 699
                local is_uninterest_depress = 0
                if remapClusterList ~= nil then
                  item_cluster632 = remapClusterList[(hetu_sim_cluster_id or 0) + 1]
                end
                for i = 1, #uninterest_cluster_list do
                  if item_cluster632 == uninterest_cluster_list[i] then
                    is_uninterest_depress = 1
                  end
                end

                return is_uninterest_depress, item_cluster632
            end
            """
          ) \
        .end_() \

  def post_process(self) -> None:
      self.flow \
        .log_debug_info(
            common_attrs = [
                "colossus_photo_id_list_new_positive", "browse_screen__pid_list", "concat_browset_cluster1000_list",
                "uninterest_cluster_list", "interest_test_k", "interest_test_v", "uninterest_test_k", "uninterest_test_v"
            ],
            item_attrs = [
                "hetu_sim_cluster_id",
                "item_cluster632",
                "is_uninterest_depress",
            ],
            for_debug_request_only = True,
            respect_sample_loggging = True,
        )