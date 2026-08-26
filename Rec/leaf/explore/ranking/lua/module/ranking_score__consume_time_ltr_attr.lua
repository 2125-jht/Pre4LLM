function calculate_changed()
    local pctr_ori = pctr or empirical_ctr
    local pctr = corr_pctr or empirical_ctr
    local pltr = pltr or empirical_ltr
    local pwtr = pwtr or empirical_wtr
    local pftr = pftr or empirical_ftr
    local psvr = psvr or 0.0
    local pcmtr = pcmtr or empirical_cmtr
    local pptr = pptr or empirical_ptr
    local pcmef = pcmef or 0.0
    local phtr = phtr or 0.0
    local pevtr = pevtr or 0.0
    local fr_score1 = fr_score1 or 0.0
    local fr_score2 = fr_score2 or 0.0
    local pdtr = pdtr or 0.0
    local fetr = fetr or 0.0
    local fountain_eff = fountain_eff or 0.0
    local cascade_pctr = cascade_pctr or 0.0
    local cascade_pltr = cascade_pltr or 0.0
    local cascade_pwtr = cascade_pwtr or 0.0
    local cascade_plvtr = cascade_plvtr or 0.0
    local cascade_psvtr = cascade_psvtr or 0.0

    local empirical_ctr = empirical_ctr or 0.0
    local empirical_ltr = empirical_ltr or 0.0
    local empirical_wtr = empirical_wtr or 0.0
    local empirical_ftr = empirical_ftr or 0.0
    local empirical_ptr = empirical_ptr or 0.0
    local empirical_cmtr = empirical_cmtr or 0.0
    local empirical_watch_time = empirical_watch_time or 0.0
    local empirical_htr = empirical_htr or 0.0

    local pdctr = pdctr or -1.0
    local pvtr = pvtr or -1.0
    local pfvtr = pfvtr or 0.0
    local plvtr = plvtr or 0.0

    local location__poi = location__poi or ""
    local music = music or ""
    local hetu_tag_level_info__hetu_face_id = hetu_tag_level_info__hetu_face_id or {}
    local hetu_tag_level_info__hetu_level_three = hetu_tag_level_info__hetu_level_three or {}
    local pShortStatShowHetu3100n = pShortStatShowHetu3100n or {}
    local pShortStatShowHetu31000n = pShortStatShowHetu31000n or {}
    local pShortStatClickHetu3100n = pShortStatClickHetu3100n or {}
    local pShortStatClickHetu31000n = pShortStatClickHetu31000n or {}
    local pShortStatClickRateHetu3100n = pShortStatClickRateHetu3100n or {}
    local pShortStatClickRateHetu31000n = pShortStatClickRateHetu31000n or {}
    local mmu_img_cluster_v3 = mmu_img_cluster_v3 or 0
    local location__city_id = location__city_id or 0
    local location__province_id = location__province_id or 0
    local audit_hot_high_tag_level = audit_hot_high_tag_level or 0
    local mmu_img_cluster_v4 = mmu_img_cluster_v4 or 0
    local hetu_tag_level_info__hetu_level_five = hetu_tag_level_info__hetu_level_five or {}
    local hetu_tag_level_info__hetu_level_two = hetu_tag_level_info__hetu_level_two or {}
    local hetu_tag_level_info__hetu_tag = hetu_tag_level_info__hetu_tag or {}
    local upload_type = upload_type or 0
    local hetu_tag_level_info__hetu_level_one = hetu_tag_level_info__hetu_level_one or {}
    local hetu_tag_level_info__hetu_cluster_id = hetu_tag_level_info__hetu_cluster_id or 0
    local total_report_count = explore_stat__report_detail__total_report_count or 0

    local infer_gender = infer_gender or 0
    local infer_year = infer_gender or 0

    return pctr, pltr, pwtr, pftr, psvr, pcmtr, pptr, pcmef, phtr, pevtr, fr_score1, fr_score2, fr_score1, fr_score2,
           pdtr, fetr, fountain_eff, cascade_pctr, cascade_pltr, cascade_pwtr, cascade_plvtr, cascade_psvtr, 1, 1,
           empirical_ctr, empirical_ltr, empirical_wtr, empirical_ftr, empirical_ptr, empirical_cmtr, empirical_htr, empirical_watch_time,
           pdctr, pvtr, pfvtr, plvtr, location__poi, music, hetu_tag_level_info__hetu_face_id, hetu_tag_level_info__hetu_level_three,
           pShortStatShowHetu3100n, pShortStatShowHetu31000n, pShortStatClickHetu3100n, pShortStatClickHetu31000n,
           pShortStatClickRateHetu3100n, pShortStatClickRateHetu31000n, mmu_img_cluster_v3, location__city_id,
           location__province_id, audit_hot_high_tag_level, mmu_img_cluster_v4, hetu_tag_level_info__hetu_level_five,
           hetu_tag_level_info__hetu_level_two, hetu_tag_level_info__hetu_tag, upload_type, hetu_tag_level_info__hetu_level_one,
           hetu_tag_level_info__hetu_cluster_id
end

function transfer_ltr()
    local time_ltr = 0.0
    if consume_time_ltr ~= nil then
        time_ltr = consume_time_ltr
    end
    local final_ltr = 1000.0 * time_ltr / math.max(1.0 - time_ltr, 1e-4)
    return final_ltr
end

function transfer_duration()
    local duration_ms_new = duration_ms or 0
    local duration = duration_ms_new / 1000
    local duration_low_0 = 0
    if (enable_sk_pftr_st_feed > 0 and duration <= 0) then
        duration_low_0 = 1
    end
    local duration_cluster = (math.floor(duration // 5)) * 5
    return duration_cluster, duration_low_0
end

function transfer_ftr()
    local consume_time_ptr_new = consume_time_ptr or 0.0
    local pftr_low_0 = 0
    if (consume_time_ptr_new < 0) then
        if(enable_sk_pftr_negative > 0) then
            pftr_low_0 = 1
        end
        consume_time_ptr_new = 0.01
    end
    consume_time_ptr_new = math.floor(consume_time_ptr_new * 100) * 1.0 / 100
    return consume_time_ptr_new, pftr_low_0
end

function transfer_key()
    local redis_key = pftr_prefx.."_"
    local duration_cluster_new = duration_cluster or 0
    local consume_time_pftr_score_new = consume_time_pftr_score or 0.0
    redis_key = redis_key..tostring(duration_cluster_new).."_"..tostring(consume_time_pftr_score_new)
    local key_tmp = tostring(consume_time_pftr_score_new)
    if (enable_modified_trunc > 0) then
        if (key_tmp == "0" or key_tmp == "1" or key_tmp == "2" or key_tmp == "3") then
            redis_key = redis_key..".0"
        end
    end
    return redis_key
end

function transfer_pf2r()
    if (duration_low_0 > 0 or pftr_low_0 > 0) then
        return 0.0
    end
    local consume_time_pf2r_new
    if(enable_zip_process > 0) then
        consume_time_pf2r_new = consume_time_ptr or 0.0
    else
        consume_time_pf2r_new = tonumber(consume_time_pf2r or "0.0")
    end
    return consume_time_pf2r_new
end
