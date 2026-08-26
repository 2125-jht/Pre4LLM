function is_mobile_enable_comment_ltr()
    local fountain_skip_cal_enrich_comment_ltr_item_attr_from_redis_all = 1
    if (fountain_skip_cal_enrich_comment_ltr_item_attr_from_redis == 0 or fountain_skip_cal_enrich_comment_ltr_item_attr_from_redis_mobile == 0) then
        fountain_skip_cal_enrich_comment_ltr_item_attr_from_redis_all = 0
    end
    local fountain_skip_cal_enrich_comment_ltr_common_attr_from_redis_all =1 
    if (fountain_skip_cal_enrich_comment_ltr_common_attr_from_redis == 0 or fountain_skip_cal_enrich_comment_ltr_common_attr_from_redis_mobile == 0) then
        fountain_skip_cal_enrich_comment_ltr_common_attr_from_redis_all = 0
    end
    local fountain_skip_cal_comment_ltr_all =1 
    if (fountain_skip_cal_comment_ltr == 0 or fountain_skip_cal_comment_ltr_mobile == 0) then
        fountain_skip_cal_comment_ltr_all = 0
    end
    local fountain_comment_ltr_model_kconf_key_all = "reco.comment.fountianCommentLtrModel"
    if (fountain_skip_cal_comment_ltr == 0) then 
        fountain_comment_ltr_model_kconf_key_all = fountain_comment_ltr_model_kconf_key
    elseif (fountain_skip_cal_comment_ltr_mobile == 0) then
        fountain_comment_ltr_model_kconf_key_all = fountain_comment_ltr_model_kconf_key_mobile
    end 
    return fountain_skip_cal_enrich_comment_ltr_item_attr_from_redis_all,fountain_skip_cal_enrich_comment_ltr_common_attr_from_redis_all,fountain_skip_cal_comment_ltr_all,fountain_comment_ltr_model_kconf_key_all
end