# -*- coding: utf-8 -*-
"""
    server2
    ~~~~~~~

    Description.

    :author: zhaoyouqing
    :copyright: (c) 2018, Tungee
    :date created: 2020-06-15
    :python version: 3.5
"""

import asyncio
import http
import json
import traceback
from time import sleep

import websockets
from websockets.http import Headers

from auth import login_auth_2, handle_user_connection
from cache import redis_inst

user_register_dict = dict()

async def notify():
    pubsub_item = redis_inst.pubsub()
    pubsub_item.subscribe('tp_msg')
    while True:
        msg = pubsub_item.get_message(ignore_subscribe_messages=True)
        if not msg:
            # print('no msg from tp')
            await asyncio.sleep(0.01)
            continue

        if not isinstance(msg, dict):
            await asyncio.sleep(0.01)
            continue

        data = msg.get('data')
        if not data:
            await asyncio.sleep(0.01)
            continue

        if isinstance(data, bytes):
            data = data.decode('utf-8')

        try:
            data = json.loads(data)
        except:
            traceback.print_exc()

            await asyncio.sleep(0.01)
            continue

        if not isinstance(data, dict):
            await asyncio.sleep(0.01)
            continue

        try:
            if user_register_dict[data['company_user_id']].open:
                await user_register_dict[data['company_user_id']].send(data['msg'])

        except websockets.exceptions.ConnectionClosed:
            traceback.print_exc()

        await asyncio.sleep(0.01)


@handle_user_connection(user_register_dict)
async def hello(websocket, path):
    while websocket.open:
        # print(path)
        # print(websocket.request_headers)
        name = await websocket.recv()
        # print(f"< {name}")

        await websocket.send('greeting')
        # await websocket.close()
        # print('111')




async def process_request(path, request_headers):
    ok, err_code, msg = login_auth_2(request_headers)
    if not ok:
        return (http.HTTPStatus.UNAUTHORIZED, Headers(),
                b"Failed to open a WebSocket connection.\n")
    request_headers['user_info'] = msg

    return None

    # print(path, request_headers, type(request_headers))
    # if isinstance(request_headers, Headers):
    #     # request_headers.
    #     print('1111')
    # return (http.HTTPStatus.UNAUTHORIZED, Headers(),
    #              b"Failed to open a WebSocket connection.\n")


start_server = websockets.serve(hello, "localhost", 10202, process_request=process_request)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_until_complete(notify())
asyncio.get_event_loop().run_forever()