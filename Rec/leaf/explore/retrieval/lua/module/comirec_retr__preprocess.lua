function prepare_request_params()
  if auxiliary_arg_list == nil or #auxiliary_arg_list < 5 then
    return nil, nil, nil, "target_tensor", "context_tensor"
  end
  return tonumber(auxiliary_arg_list[1]), tonumber(auxiliary_arg_list[2]), tonumber(auxiliary_arg_list[3]), auxiliary_arg_list[4], auxiliary_arg_list[5]
end