"""Strict standard-library float32 codec; legacy official float responses kept."""
import asyncio
import base64
import json
import math
import socket
import struct

import httpx
import pytest

from core import qwen_embedding as e


def encoded(values):return base64.b64encode(struct.pack('<'+str(len(values))+'f',*values)).decode('ascii')


def payload(vector):return json.dumps({'model':e.MODEL,'data':[{'index':0,'embedding':vector}]}).encode()


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*a,**k):raise AssertionError('real network forbidden')
    monkeypatch.setattr(socket.socket,'connect',denied)
    monkeypatch.setattr(socket.socket,'connect_ex',denied)
    monkeypatch.setattr(socket,'create_connection',denied)
    monkeypatch.setenv('ECHOMIND_ENV','development')


def test_base64_and_documented_float_gateway_response_have_same_normalized_values():
    floats=[3.,4.]+[0.]*1022
    assert e.parse_vectors(payload(encoded(floats)),1)==e.parse_vectors(payload(floats),1)==[[.6,.8]+[0.]*1022]


@pytest.mark.parametrize('damage',['invalid_char','short','long','nan','inf','zero','unicode','padding'])
def test_invalid_or_nonfinite_base64_is_rejected_without_repair(damage):
    values=[1.]+[0.]*1023
    if damage=='nan':values[9]=float('nan')
    elif damage=='inf':values[9]=float('inf')
    elif damage=='zero':values=[0.]*1024
    value=encoded(values)
    if damage=='invalid_char':value='!'+value[1:]
    elif damage=='short':value=encoded(values[:-1])
    elif damage=='long':value=encoded(values+[0.])
    elif damage=='unicode':value='中'+value[1:]
    elif damage=='padding':value=value[:-2]+'AA'
    with pytest.raises(e.EmbeddingError,match='response_invalid'):e.parse_vectors(payload(value),1)


def test_mixed_formats_wrong_indices_and_duplicate_indices_remain_rejected():
    rows=[{'index':0,'embedding':encoded([1.]*1024)},{'index':1,'embedding':[1.]*1024}]
    with pytest.raises(e.EmbeddingError):e.parse_vectors(json.dumps({'data':rows}),2)
    for bad in (True,1,-1):
        with pytest.raises(e.EmbeddingError):
            e.parse_vectors(json.dumps({'data':[{'index':bad,'embedding':encoded([1.]*1024)}]}),1)


@pytest.mark.parametrize('returned_format',['float','base64'])
def test_production_requests_compact_format_and_accepts_only_valid_full_vectors(monkeypatch,returned_format):
    async def run():
        calls=[]
        def provider(request):
            calls.append(request)
            assert json.loads(request.content)=={'model':e.MODEL,'dimensions':1024,'encoding_format':'base64','input':['q']}
            values=[3.,4.]+[0.]*1022
            return httpx.Response(200,content=payload(encoded(values) if returned_format=='base64' else values))
        monkeypatch.setattr(e,'_delegate',lambda:httpx.MockTransport(provider))
        diagnostic={}
        vectors=await e.request_embeddings(['q'],base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
            api_key='controlled-offline-key',diagnostic=diagnostic)
        assert len(calls)==1 and vectors[0][:2]==[.6,.8]
        assert diagnostic['phase']=='complete' and diagnostic['error_code'] is None
    asyncio.run(run())


@pytest.mark.parametrize('fmt',['invalid',None,[],True])
def test_invalid_internal_codec_is_not_silently_replaced(fmt):
    with pytest.raises(e.EmbeddingError,match='input_invalid'):e._payload(['q'],encoding_format=fmt)


def test_vector_bits_are_little_endian_and_nontrivial_values_stay_finite():
    values=[.5,-.25,1.25,-2.]+[.001]*1020
    binary=base64.b64decode(encoded(values),validate=True)
    assert binary[:4]==b'\x00\x00\x00?'
    result=e.parse_vectors(payload(encoded(values)),1)[0]
    assert result[0]>0 and result[1]<0 and result[2]>result[0] and result[3]<result[1]
    assert math.isclose(math.hypot(*result),1.,abs_tol=1e-12)
