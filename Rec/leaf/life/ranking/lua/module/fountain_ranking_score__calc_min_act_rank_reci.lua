function calc_min_act_rank_reci()
    local min_act_rank = 500
    local act_weights_tb = fountain_fullrank_min_act_rank_weights and fountain_fullrank_min_act_rank_weights or {}
    local act_rank_tb = {fullrank_like_rank, fullrank_cmtr_rank, fullrank_cmef_rank, fullrank_lstr_rank, 
                            fullrank_epstr_rank, fullrank_follow_rank, fullrank_ftr_rank, fullrank_cltr_rank}
    if (#act_weights_tb == #act_rank_tb) then
        for i = 1, #act_rank_tb do
            if (act_weights_tb[i] ~= nil and act_weights_tb[i] > 0) then
                min_act_rank = math.min(min_act_rank, act_rank_tb[i])
            end
        end
    end
    min_act_rank = 1.0 / (min_act_rank + 10.0)
    return min_act_rank
end

function calc_min_wt_rank_reci()
    local min_wt_rank = 500
    local wt_weight_tb = fountain_fullrank_min_wt_rank_weights and fountain_fullrank_min_wt_rank_weights or {}
    local wt_rank_tb = {
        fullrank_trans_pvtr_rank, 
        fullrank_sim_pvtr_rank,
        fullrank_sim_pevtr_rank,
        fullrank_detail_new_pevtr_v2_rank,
        fullrank_sim_plvtr_rank,
        fullrank_sim_pfintr_rank,
        fullrank_ltr_v4_fountain_finish_rate_rank,
        fullrank_ltr_v4_fountain_next_rank,
    }
    if (#wt_weight_tb == #wt_rank_tb) then
        for i = 1, #wt_weight_tb do
            if (wt_weight_tb[i] ~= nil and wt_weight_tb[i] > 0.0 and wt_rank_tb[i] ~= nil) then
                min_wt_rank = math.min(wt_weight_tb[i]*wt_rank_tb[i], min_wt_rank)
            end
        end
    end
    min_wt_rank = 1.0 / (min_wt_rank + 10.0)
    return min_wt_rank
end