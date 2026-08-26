-- 计算splash ctr ensemble sort weight
function cal_splash_ctr_sort_weight()
    local splash_ctr_sort_weight = 0.0
    if (skip_fountain_splash_ctr_predict == 0 and common_request_type == fountain_splash_request_type) then
        splash_ctr_sort_weight = fountain_splash_fullrank_variant_weight2_splash_ctr
    end
    return splash_ctr_sort_weight
end

function is_skip_fountain_splash_ctr_predict()
    local is_skip_fountain_splash_ctr_predict = 1
    if (skip_fountain_splash_ctr_predict == 0 and common_request_type == fountain_splash_request_type) then
        is_skip_fountain_splash_ctr_predict = 0
    end
    return is_skip_fountain_splash_ctr_predict
end
