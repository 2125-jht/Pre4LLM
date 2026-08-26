function pure_value_es_value_score()
    local alpha = fullrank_pure_value_es_value_alpha or 0.0
    local beta = fullrank_pure_value_es_value_beta or 0.0
    local bias = fullrank_pure_value_es_value_bias or 1.0
    local es_score = fullrank_ensemble_score or 0.0
    local pv_score = fullrank_pure_value_score or 0.0
    if (fullrank_pure_value_es_value_use_multiply == 1) then
        es_score = es_score * ( bias + alpha * (pv_score ^ beta))
    elseif (fullrank_pure_value_es_value_use_multiply == 2) then
        es_score = es_score * ((bias + alpha * pv_score) ^ beta)
    else
        es_score = alpha * es_score + beta * pv_score
    end

    return es_score
end