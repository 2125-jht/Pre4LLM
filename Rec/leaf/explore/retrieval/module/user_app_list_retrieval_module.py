from retrieval.retrieval_module import RetrievalModule

class UserAppListRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .if_(
        "(enable_explore_applist_retr_divide_active_adjust == 1 and (find_user_active_degree ~= 3 and find_user_active_degree ~= 4)) or "
        "(enable_explore_applist_retr_divide_vv_adjust == 1 and (active_days_avg_vv <= explore_applist_retr_divide_vv_threshold)) or "
        "(enable_explore_applist_retr_divide_active_adjust ~= 1 and enable_explore_applist_retr_divide_vv_adjust ~= 1)"
      ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "app_list_ios",
            {"name": "app_list", "as": "app_list_android"},
            {"name": "enable_explore_user_applist_random_interested_cids", "as": "enable_random_interested_cids"},
            {"name": "ios_app_name_cluster_id_map_ptr", "as": "ios_app_list_interested_cids"},
            {"name": "android_app_package_id_cluster_id_map_ptr", "as": "android_app_list_interested_cids"},
          ],
          export_common_attr = [
            "user_app_list_interested_cids",
          ],
          function_name = "GetUserAppListInterestedCids",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .retrieve_by_remote_index(
          kess_service = "{{cluster_remote_index_service_name}}",
          timeout_ms = 25,
          reason = self.reason, 
          querys = [
            {
              "query": "cid:{{user_app_list_interested_cids}}",
              "search_num": "{{cid_index_search_num}}", 
              "max_attr_num": "{{retrieve_num}}"
            }
          ]
        ) \
        .limit(
          size = "{{cand_num}}"
        ) \
      .end_() 
