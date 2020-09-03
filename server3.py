# -*- coding: utf-8 -*-
"""
    server3
    ~~~~~~~

    Description.

    :author: zhaoyouqing
    :copyright: (c) 2018, Tungee
    :date created: 2020-06-16
    :python version: 3.5
"""
from typing import Union, Optional, Awaitable, Any

import tornado.ioloop
import tornado.web
import tornado.websocket
from itsdangerous import URLSafeTimedSerializer, BadSignature
from tornado import httputil
from tornado.httpserver import HTTPServer

from auth import _auth_user_


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.send_error()
        self.write_error(400, msg='sdaf')

    def post(self):
        company_user_id = self.get_body_argument('company_user_id')
        msg = self.get_body_argument('msg')
        hit_item = user_register_dict.get(company_user_id)
        if not hit_item:
            self.send_error(400)
            return None

        if not isinstance(hit_item, WSHandler):
            self.send_error(400)
            return None

        hit_item.write_message(msg)

        self.finish({'stat': 1})

    def write_error(self, status_code: int, **kwargs: Any) -> None:
        self.finish(kwargs)


class UserAuth(object):

    def __init__(self) -> None:
        super().__init__()
        self.__secret_key__ = '000000'
        self.canary_key = 'canary:'

    def auth(self):
        raise NotImplementedError

    def get_session(self, cookie_val, salt='cookie-session'):
        s = URLSafeTimedSerializer(self.__secret_key__, salt)

        # 获取账号中心的session
        account_val = cookie_val

        if not isinstance(account_val, str):
            return 0, 1, None

        # 去掉sessionId的金丝雀用户标记
        if account_val.startswith(self.canary_key):
            account_val = account_val[len(self.canary_key):]

        if not account_val:
            return 0, 1, None
        try:
            account_data = s.loads(account_val)
            fields = ['sid', 'channel']
            session = {}
            for field in fields:
                if field not in account_data:
                    continue

                session[field] = account_data.get(field)
        except BadSignature:
            return 0, 2, '解析session失败'

        return 1, 0, session


user_register_dict = dict()


class WSHandler(tornado.websocket.WebSocketHandler, UserAuth):

    def __init__(self, application: tornado.web.Application,
                 request: httputil.HTTPServerRequest, **kwargs: Any) -> None:
        super().__init__(application, request, **kwargs)
        self.company_id = None
        self.company_user_id = None

    def auth(self):
        cookie_val = self.get_cookie('accountCenterSessionId')
        if not cookie_val:
            return 0, 1, 'no cookie'
        ok, err_code, msg = self.get_session(cookie_val)
        if not ok:
            return ok, err_code, msg
        ok, err_code, msg = _auth_user_(msg)
        if not ok:
            return ok, err_code, msg

        return 1, 0, msg

    def prepare(self) -> Optional[Awaitable[None]]:
        ok, err_code, msg = self.auth()
        print(ok, err_code, msg)
        if not ok:
            self.write_error(401, msg=msg)
            return None

        self.company_id = msg['company_id']
        self.company_user_id = msg['company_user_id']
        return None

    def open(self, *args: str, **kwargs: str) -> Optional[Awaitable[None]]:
        # session_val = self.request.cookies.get('accountCenterSessionId')
        # if not isinstance(session_val, Morsel):
        # self.write_error(401, msg={'sdf': 'fsdaf'})
        if self.company_user_id is None:
            self.write_error(401)
            return None

        if self.company_user_id in user_register_dict:
            exist_ws_handler = user_register_dict[self.company_user_id]
            if isinstance(exist_ws_handler, WSHandler) and \
                    not exist_ws_handler.ws_connection.is_closing():
                try:
                    exist_ws_handler.close()
                except:
                    pass

        user_register_dict[self.company_user_id] = self
        return None

    def on_message(self, message: Union[str, bytes]) -> Optional[
            Awaitable[None]]:
        self.write_message(message)
        return None

    def on_finish(self) -> None:
        if self.company_user_id is None:
            return None

        if user_register_dict.get(self.company_user_id) != self:
            return None

        user_register_dict.pop(self.company_user_id)

        return None


def make_app():
    return tornado.web.Application(
        [
            (r"/tp-api/notify", MainHandler),
            (r'/ws/', WSHandler)
        ],
        debug=True,
        autoreload=True,
        websocket_ping_interval=1
    )

if __name__ == '__main__':
    app = make_app()
    server = HTTPServer(app)
    server.listen(10202, 'localhost')
    tornado.ioloop.IOLoop.current().start()
