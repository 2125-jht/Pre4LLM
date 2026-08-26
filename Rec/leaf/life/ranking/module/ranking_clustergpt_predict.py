from ranking import CommonModule


class RankingClustergptPredictModule(CommonModule):
    def process(self) -> None:
        self.flow\
            .if_("life_enable_frs2_clustergpt == 1") \
                .delegate_enrich(
                    kess_service = "{{life_frs2_clustergpt_service}}",
                    send_common_attrs = [{"name": "userInfo", "as": "user_info_str"},
                                         {"name": "week_cluster_list", "as": "week_cluster_list"}],
                    send_item_attrs = [],
                    recv_common_attrs = [{"name": "all_cluster_like_prob", "as": "longterm_all_cluster_like_prob"}],
                    request_type = "{{life_frs2_clustergpt_request_type}}",
                    timeout_ms = "{{life_frs2_clustergpt_timeout_ms}}",
                ) \
                .get_kconf_params(
                    kconf_configs = [{
                    "kconf_key": "reco.interestExplore.remapClusterId632",
                    "export_common_attr": "remapClusterList",
                    "value_type": "list_int64",
                    }]
                ) \
                .enrich_attr_by_lua(
                    import_common_attr=["remapClusterList", "week_cluster_list"],
                    import_item_attr=["hetu_sim_cluster_id"],
                    export_item_attr=["is_out_week", "map_632cluster"],
                    function_for_item="calculate",
                    lua_script="""
                    function calculate()              
                        if remapClusterList == nil  then
                        return 0.0, 0
                        end
                        
                        local mapping_cluster = remapClusterList[(hetu_sim_cluster_id or 0)+1]              
                        if week_cluster_list == nil then
                        return 1.0, mapping_cluster
                        end 
                        
                        -- 判断茧房内样本
                        local is_out_cocoon = 1.0
                        for i=1, #week_cluster_list do
                        if (mapping_cluster == week_cluster_list[i]) then
                            is_out_cocoon = 0.0
                            break
                        end
                        end
                        return is_out_cocoon * 1.0, mapping_cluster
                    end
                    """
                )\
                .enrich_attr_by_lua(
                    import_common_attr = ['longterm_all_cluster_like_prob'],
                    import_item_attr = ['map_632cluster'],
                    export_item_attr = ['longterm_cluster_score'],
                    function_for_item = 'get_longterm_cluster_score',
                    lua_script = """
                    function get_longterm_cluster_score()
                        local xs = longterm_all_cluster_like_prob
                        if xs == nil or map_632cluster >= #xs then
                            return 0.0
                        end
                        return xs[map_632cluster+1]
                    end
                    """
                )\
                .if_("life_enable_f1_fr_longterm_cluster_score == 1") \
                    .calc_by_formula1(
                        kconf_key = "formula.scenarioKey12.LifeMcLongtermClusterScore",
                        import_item_attr = [
                        "longterm_cluster_score",
                        "is_out_week",
                        "pctr", 
                        "pltr", 
                        "pevtr",
                        "plvtr", 
                        "psvr"
                        ],
                        export_formula_value = [
                        "longterm_cluster_score"
                        ],
                        abtest_biz_name = "KUAISHOU_APPS"
                    ) \
                .end_() \
            .end_() \

    def post_process(self) -> None:
        self.flow \
        .log_debug_info(
            common_attrs = [
            "week_valid_cluster_redis_key", "week_cluster_list", 'longterm_all_cluster_like_prob'
            ],
            item_attrs = [
            "photo_id", "longterm_cluster_score", "hetu_sim_cluster_id", "hetu_tag_level_info__hetu_cluster_id", 'map_632cluster', "is_out_week",
            "pctr", 'pltr', "pevtr", "plvtr", 'psvr'
            ],
            for_debug_request_only = True,
        )