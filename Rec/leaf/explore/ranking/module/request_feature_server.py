from ranking import CommonModule
from core.base_reco_flow import BaseRecoFlow
from dragonfly.decorators import if_

'''
请求 feature server, 用于离线特征抽取
'''


class RequestFeatureServerModule(CommonModule):

    def owner(self) -> str:
        return 'xudongyi'

    def process(self):

        self.flow.gen_common_attr_by_lua(
            attr_map={
                "fs_flow_control_id": "util.CityHash64(request_id)",
            }
        )
        self.flow.check_tail_number(
            kconf_key='reco.arch.enableExploreRequestFs',
            test_value='{{fs_flow_control_id}}',
            output_to='enable_request_feature_server',
        )

        self.request_fs(self.flow)

    @if_('enable_request_feature_server == 1 and (fr_sample_discard or 0) == 0')
    def request_fs(self, flow: BaseRecoFlow):
        flow.set_attr_value(
            common_attrs=[dict(name='fs_caller_biz', type='string', value='explore'),
                          dict(name='fs_caller_stage',
                               type='string', value='ranking')
                          ]
        )
        flow.str_format(
            format_string="%s%s",
            input_attrs=["request_id", "device_id"],
            output_attr="feature_server_hash_id",
        )

        flow.delegate_enrich(
            kess_service="reco-feature-server-ranking",
            request_type="explore-ranking-infer",
            consistent_hash=True,
            hash_id="{{feature_server_hash_id}}",
            send_item_attrs=[
                # FS 必需参数
                {"name": "reco_photo_info_str", "as": "context_info_str"},
                {"name": "reason_str", "as": "_reason"},
                {"name": "live_photo_info__is_living", "as": "_is_living"},

                {"name": "cascade_pctr_index", "as": "cascade_pctr_index"},
                {"name": "cascade_plvtr_index", "as": "cascade_plvtr_index"},
                {"name": "cascade_pvtr_index", "as": "cascade_pvtr_index"},
                {"name": "cascade_pltr_index", "as": "cascade_pltr_index"},
                {"name": "cascade_pftr_index", "as": "cascade_pftr_index"},
                {"name": "cascade_pwtr_index", "as": "cascade_pwtr_index"},
            ],
            send_common_attrs=[
                {"name": "fs_caller_biz", "as": "caller_biz"},
                {"name": "fs_caller_stage", "as": "caller_stage"},

                # FS 必需参数
                {"name": "virtualTabId", "as": "_tab_id"},
                {"name": "userInfo", "as": "user_info_str"},
                {"name": "page_index", "as": "page_common"},

                # 目前仅用于离线抽特征，以下暂时不需要
                # {"name": "", "as": "infer_kess_service"},
                # {"name": "", "as": "infer_request_type"},
                # {"name": "", "as": "infer_partition_size"},
                # {"name": "", "as": "infer_timeout_ms"},
                # {"name": "", "as": "enable_infer_via_fs"},

            ],
            recv_common_attrs=["fs_res"],  # 无实际作用，仅用于成功请求
            use_packed_item_attr=True,
        )
