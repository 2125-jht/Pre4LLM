function adjust_quota_size(size, factor, increase_quota_status)
    local factor = factor and factor or 1.0
    local increase_quota_status = increase_quota_status and increase_quota_status or 0
    local size = size
    if (increase_quota_status > 0) then
        size = math.floor(size * factor)
    end
    return size
end

function adjust_pre_filter_photo_size()
    return adjust_quota_size(fullrank_splash_pre_filter_keep_photo_size, fullrank_splash_pre_filter_keep_photo_size_increase_quota_factor, increase_quota_status)
end