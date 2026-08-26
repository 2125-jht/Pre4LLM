function change_sl_interactive_score()
    local sl_interactive_score = fountain_ensemble_weight_fullrank_sl_interactive_score or 0.0
    local fountain_sl_only_fast_v1 = fountain_sl_only_fast_v1 or 0
    local page = page or 1

    if (fountain_sl_only_fast_v1 == 0 or (fountain_sl_only_fast_v1 == 1 and page > 1)) then
        return sl_interactive_score
    end

    return 0.0
end

function change_pure_value_score()
    local pure_value_score = fountain_ensemble_weight_fullrank_pure_value_score or 0.0
    local pv_es_rank = skip_fullrank_pure_value_es_rank_score or 1
    local pv_es_value = skip_fullrank_pure_value_es_value_score or 1
    local fountain_pure_value_only_fast_v1 = fountain_pure_value_only_fast_v1 or 0
    local page = page or 1

    if (fountain_pure_value_only_fast_v1 == 0 or (fountain_pure_value_only_fast_v1 == 1 and page > 1)) then
        return pure_value_score, pv_es_rank, pv_es_value
    end

    return 0.0, 1, 1
end
