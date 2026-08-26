from ranking import CommonModule

class FountainFetchSimilarUserListModule(CommonModule):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def process(self) -> None:
      self.flow \
        .enrich_attr_by_lua(
          import_common_attr = [
            "featureUId",
            "fountain_similar_user_list_redis_prefix",
          ],
          export_common_attr = [
            "fetch_similar_user_list_redis_key"
          ],
          function_for_common = "gen_similar_users_redis_key",
          lua_script_file = "life/ranking/lua/module/fountain_ranking_score__trans_reason_to_str.lua"
        ) \
        .get_common_attr_from_redis(
          cluster_name = "recoColossusTriggers",
          redis_params = [
            {
              "redis_key": "{{fetch_similar_user_list_redis_key}}",
              "output_attr_name": "featureSimilarUserList_str"
            }
          ]
        ) \
        .split_string(
          input_common_attr = "featureSimilarUserList_str",
          output_common_attr = "featureSimilarUserList_str_list",
          delimiters=",",
        ) \
        .merchant_split_string_list(
          input_common_attr = "featureSimilarUserList_str_list",
          output_attr_configs = [
            {
              "export_common_attr": "featureSimilarUserList",
              "parse_to_type": "list_int64",
              "pos_in_splitted": 0,
              "default_value": "-1"
            },
            {
              "export_common_attr": "featureSimilarUserScoreList",
              "parse_to_type": "list_double",
              "pos_in_splitted": 1,
              "default_value": "0.0"
            }],
          delimiters=":"
        )

    def post_process(self) -> None:
      self.flow \
        .log_debug_info(
          common_attrs = [
            "featureSimilarUserList",
            "featureSimilarUserScoreList"
          ],
          item_num_limit = 10,
          for_debug_request_only = True,
        )
