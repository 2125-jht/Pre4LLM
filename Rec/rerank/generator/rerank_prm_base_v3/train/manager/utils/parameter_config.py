# -*- coding: UTF8 -*-
import os
import sys
import json


def attr_names_by_channel(attr_name):
    channel_postfix = ['Hot', 'Follow', 'Near', '']
    return [attr_name + c for c in channel_postfix]


class ExtractorType:
    EMBEDDING_PARAMATER = 1
    LR_PARAMETER = 2
    THIRD_PARTY_DATA = 3
    FLAT_EMBEDDING_PARAMATER = 5


def customDiscreteExtractor(attrs, slot_id, discrete_converter_args="1,2,3",
                            extractor_type=ExtractorType.EMBEDDING_PARAMATER, cache_shard=""):
    assert isinstance(attrs, (list, str))
    attrs = [attrs] if isinstance(attrs, str) else attrs
    return dict(
        type=extractor_type,
        attrs=[
            dict(attr=attrs, key_type=slot_id, converter='custom_discrete', converter_args=discrete_converter_args,
                 cache_shard=cache_shard),
        ],
        dim=64,
        batch_num=10,
        batch_decay=0.98,
    )


def floatDiscreteExtractor(attrs, discrete_converter_args, key_type=0,
                           extractor_type=ExtractorType.EMBEDDING_PARAMATER, cache_shard="", batch_num=None,
                           batch_decay=None, expire_second=None, dim=None, use_common_attr_only=None):
    assert isinstance(attrs, (list, str))
    attrs = [attrs] if isinstance(attrs, str) else attrs
    fea = dict(
        type=extractor_type,
        attrs=[
            dict(attr=attrs, key_type=key_type, converter='discrete', converter_args=discrete_converter_args,
                 cache_shard=cache_shard),
        ],
    )
    if dim:
        fea['dim'] = dim
    if batch_num:
        fea['batch_num'] = batch_num
    if batch_decay:
        fea['batch_decay'] = batch_decay
    if expire_second:
        fea['expire_second'] = expire_second
    if use_common_attr_only:
        fea['use_common_attr_only'] = use_common_attr_only
    return fea


def kuibaCombineExtractor(l_attrs_dict, r_attrs_dict, slot_id,
                          extractor_type=ExtractorType.EMBEDDING_PARAMATER):
    attr_dicts = []
    attr_dicts.append(
        dict(
            key_type=slot_id,
            converter="combine",
            converter_args=dict(
                left=l_attrs_dict,
                right=r_attrs_dict
            )
        )
    )
    for (k, v) in l_attrs_dict.items():
        assert (v >= 0 and v < 4)
    for (k, v) in r_attrs_dict.items():
        assert (v >= 0 and v < 4)

    return dict(
        type=extractor_type,
        attrs=attr_dicts,
        dim=64,
        batch_num=10,
        batch_decay=0.98
    )


def matchExtractor(l_attrs, r_attrs, slot_id,
                   extractor_type=ExtractorType.EMBEDDING_PARAMATER):
    left_attrs = [l_attrs] if not isinstance(l_attrs, list) else l_attrs
    right_attrs = [r_attrs] if not isinstance(r_attrs, list) else r_attrs
    attr_dicts = []
    for left_attr in left_attrs:
        for right_attr in right_attrs:
            attrs = [left_attr, right_attr]
            attr_dicts.append(dict(attr=attrs, key_type=slot_id, converter='match'))
    return dict(
        type=extractor_type,
        attrs=attr_dicts,
        dim=64,
        batch_num=10,
        batch_decay=0.98
    )


def distanceExtractor(attrs, slot_id, discrete_converter_args="10,0.01,1000000,1",
                      extractor_type=ExtractorType.EMBEDDING_PARAMATER):
    assert isinstance(attrs, list)
    return dict(
        type=extractor_type,
        attrs=[
            dict(attr=attrs, key_type=slot_id, converter='distance', converter_args=discrete_converter_args),
        ],
        dim=64,
        batch_num=10,
        batch_decay=0.98,
    )


def listExtractor(attrs, key_type=0, join_limit=200, extractor_type=ExtractorType.EMBEDDING_PARAMATER, cargs=None,
                  cache_shard="",
                  batch_num=None, batch_decay=None, expire_second=None, dim=None, use_common_attr_only=None):
    assert isinstance(attrs, (list, str))
    attrs = [attrs] if isinstance(attrs, str) else attrs
    fea = dict(
        type=extractor_type,
        attrs=[
            dict(attr=attrs, key_type=key_type, converter='list', converter_args=cargs, cache_shard=cache_shard),
        ],
        join_limit=join_limit
    )
    if dim:
        fea['dim'] = dim
    if batch_num:
        fea['batch_num'] = batch_num
    if batch_decay:
        fea['batch_decay'] = batch_decay
    if expire_second:
        fea['expire_second'] = expire_second
    if use_common_attr_only:
        fea['use_common_attr_only'] = use_common_attr_only
    return fea


def matrixExtractor(attrs, key_type=0, extractor_type=ExtractorType.FLAT_EMBEDDING_PARAMATER, cargs=None, cache_shard="",
                    batch_num=1, batch_decay=0.9999, expire_second=86400 * 30, dim=16, join_limit=200,initial_lr=None):
    assert isinstance(attrs, (list, str))
    attrs = [attrs] if isinstance(attrs, str) else attrs
    res = dict(
        type=extractor_type,
        attrs=[
            dict(attr=attrs, key_type=key_type, converter='list', converter_args=cargs, cache_shard=cache_shard),
        ],
        dim=dim,
        batch_num=batch_num,
        batch_decay=batch_decay,
        expire_second=expire_second,
        join_limit=join_limit
    )
    if initial_lr:
        res["initial_lr"] = initial_lr
    return res


def crossExtractor(attrs, key_type=0, extractor_type=ExtractorType.EMBEDDING_PARAMATER, dim=None, batch_num=None,
                   batch_decay=None, expire_second=None):
    assert isinstance(attrs, (list, str))
    attrs = [attrs] if isinstance(attrs, str) else attrs
    fea = dict(
        type=extractor_type,
        attrs=[
            dict(attr=attrs,
                 key_type=key_type,
                 converter='cross',
                 converter_args=dict(
                     need_dulp=True
                 )
                 ),
        ],
    )
    if dim:
        fea['dim'] = dim
    if batch_num:
        fea['batch_num'] = batch_num
    if batch_decay:
        fea['batch_decay'] = batch_decay
    if expire_second:
        fea['expire_second'] = expire_second
    return fea


def combineExtractor(info_json, slot_id, extractor_type=ExtractorType.EMBEDDING_PARAMATER):
    # assert isinstance(attrs, (list, str))
    # attrs = [attrs] if isinstance(attrs, str) else attrs
    return dict(
        type=extractor_type,
        attrs=[
            # dict(attr=attrs, key_type=slot_id, converter='combine'),
            dict(converter_args=info_json, key_type=slot_id, converter='combine'),
        ],
        dim=1,
        batch_num=1,
        batch_decay=0.98,
    )


def intExtractor(attrs, key_type=0, extractor_type=ExtractorType.EMBEDDING_PARAMATER, dim=None, cache_shard=None,
                 batch_num=None,
                 batch_decay=None, expire_second=None, use_common_attr_only=None,initial_lr=None):
    """
    :param attrs:
    :param key_type:
    :param extractor_type:
    :param dim:
    :param cache_shard:
    :param batch_num:
    :param batch_decay:
    :param expire_second:
    :return:
    key_type = 0 已废弃，为了兼容老版关注流，暂时不删除
    """
    assert isinstance(attrs, (list, str))
    attrs = [attrs] if isinstance(attrs, str) else attrs
    fea = dict(
        type=extractor_type,
        attrs=[
            dict(
                attr=attrs,
                key_type=key_type,
                converter='id',
                # cache_shard=cache_shard
            ),
        ],
    )
    if dim:
        fea['dim'] = dim
    if batch_num:
        fea['batch_num'] = batch_num
    if batch_decay:
        fea['batch_decay'] = batch_decay
    if expire_second:
        fea['expire_second'] = expire_second
    if use_common_attr_only:
        fea['use_common_attr_only'] = use_common_attr_only
    if initial_lr:
        fea['initial_lr'] = initial_lr
    return fea

def numericExtractor(attrs, key_type=0, dim=None, cache_shard=None,
                 batch_num=None,
                 batch_decay=None, expire_second=None, use_common_attr_only=None):
    """
    :param attrs:
    :param key_type:
    :param extractor_type:
    :param dim:
    :param cache_shard:
    :param batch_num:
    :param batch_decay:
    :param expire_second:
    :return:
    key_type = 0 已废弃，为了兼容老版关注流，暂时不删除
    """
    assert isinstance(attrs, (list, str))
    attrs = [attrs] if isinstance(attrs, str) else attrs
    fea = dict(
        attrs=[
            dict(
                attr=attrs,
                key_type=key_type,
                converter='numeric',
                # cache_shard=cache_shard
            ),
        ],
    )
    if dim:
        fea['dim'] = dim
    if batch_num:
        fea['batch_num'] = batch_num
    if batch_decay:
        fea['batch_decay'] = batch_decay
    if expire_second:
        fea['expire_second'] = expire_second
    if use_common_attr_only:
        fea['use_common_attr_only'] = use_common_attr_only
    return fea
