function calculate()
    local normalized_item_seq = 1.0
    if (item_seq_current ~= nil and item_seq_current >= 0) then
        normalized_item_seq = 1.0 / (item_seq_current + 10)
    end
    return normalized_item_seq
end
