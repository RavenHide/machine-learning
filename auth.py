# -*- coding: utf-8 -*-
"""
    auth
    ~~~~~~~

    Description.

    :author: zhaoyouqing
    :copyright: (c) 2018, Tungee
    :date created: 2020-06-15
    :python version: 3.5
"""
import re
from functools import wraps

from flask import request, current_app
from geventwebsocket.websocket import WebSocket
from itsdangerous import URLSafeTimedSerializer, BadSignature
from websockets import WebSocketServerProtocol

from account_model.company import Company
from account_model.company_user import CompanyUser
from account_model.individual_user import IndividualUser
from account_model.session import Session

canary_key = 'canary:'


def login_auth(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        app = current_app
        cookie_val = __get_session_val__('accountCenterSessionId')
        ok, err_code, msg = _get_session_(app.secret_key, cookie_val)
        if not ok:
            return None
        ok, err_code, msg = _auth_user_(msg)
        if not ok:
            return None

        web_socket_inst = None
        for _ in args:
            if not isinstance(_, WebSocket):
                continue
            kwargs.update(msg)

            web_socket_inst = _
            break

        if not web_socket_inst:
            for _ in kwargs.values():
                if not isinstance(_, WebSocket):
                    continue

                kwargs.update(msg)

                web_socket_inst = _
                break

        if not web_socket_inst:
            raise RuntimeError('没有 Web socket实例')

        company_user_id = msg['company_user_id']
        user_socket_dict = app.extensions['user_socket_dict']
        if company_user_id in user_socket_dict:
            try:
                user_socket_dict[company_user_id].close()
            except:
                pass

        user_socket_dict[company_user_id] = web_socket_inst

        try:
            rtn_data = func(*args, **kwargs)
        except Exception as e:
            raise e
        finally:
            user_socket_dict.pop(company_user_id, None)

        return rtn_data

    return wrapper


def __get_session_val__(session_cookie_name):
    val = request.cookies.get(session_cookie_name) \
          or request.headers.get(session_cookie_name) \
          or request.form.get(session_cookie_name) \
          or request.args.get(session_cookie_name)
    return val


def _auth_user_(session_obj):
    """
    校验用户
    :param session:
    :return:
    """
    if 'sid' not in session_obj:
        return 0, 1, '没有sid'

    session = Session.s_col.find_one({'_id': session_obj['sid']})
    if not session:
        return 0, 1, 'sid无效'

    if session.get('has_been_crowded_out', False):
        return 0, 1, '已被挤出'

    indi_user_id = session.get(Session.Field.indi_user_id, '')
    indi_user = IndividualUser.s_col.find_one(
        {'_id': indi_user_id},
        []
    )
    if not indi_user:
        return 0, 2, '用户不存在'

    company_id = session.get(Session.Field.company_id)
    if not company_id:
        return 0, 2, '公司id不存在'

    company_user = CompanyUser.s_col.find_one(
        {
            CompanyUser.Field.indi_user_id: indi_user_id,
            CompanyUser.Field.company_id: company_id,
            CompanyUser.Field.status: {'$ne': CompanyUser.Status.deleted}
        },
        []
    )

    if not company_user:
        return 0, 3, '企业用户不存在'

    company = Company.s_col.find_one(
        {'_id': company_id},
        [
            Company.Field.status
        ]
    )
    if not company:
        return 0, 4, '企业不存在'

    status = company.get(Company.Field.status)
    if status != Company.Status.normal:
        return 0, 5, '企业状态非正常状态'

    rtn_data = {
        'company_id': company_id,
        'company_user_id': company_user['_id']
    }

    return 1, 0, rtn_data


def _get_session_(secret_key, cookie_val, salt='cookie-session'):
    """
    获取session
    :param app:
    :param salt:
    :param key:
    :returns:
        ok: 1 or 0
        err_code:
            1 - session为空
        msg:
    """
    s = URLSafeTimedSerializer(secret_key, salt)

    # 获取账号中心的session
    account_val = cookie_val

    if not account_val:
        return 0, 1, None

    # 去掉sessionId的金丝雀用户标记
    if account_val.startswith(canary_key):
        account_val = account_val[len(canary_key):]

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


_cookie_re = re.compile("""(?x)
    (?P<key>[^=]+)
    \s*=\s*
    (?P<val>
        "(?:[^\\\\"]|\\\\.)*" |
         (?:.*?)
    )
    \s*;
""")


def _get_cookies_(cookie_str):
    cookie_dict = dict()
    if not isinstance(cookie_str, str):
        return cookie_dict

    i = 0
    n = len(cookie_str)

    while i < n:
        match = _cookie_re.search(cookie_str + ';', i)
        if not match:
            break

        key = match.group('key').strip()
        value = match.group('val')
        i = match.end(0)
        cookie_dict[key] = value

    return cookie_dict


def login_auth_2(request_headers):
    cookie_dict = _get_cookies_(request_headers.get('Cookie'))
    secret_key = '000000'
    ok, err_code, msg = _get_session_(
        secret_key, cookie_dict.get('accountCenterSessionId')
    )
    if not ok:
        return ok, err_code, msg
    ok, err_code, msg = _auth_user_(msg)
    if not ok:
        return ok, err_code, msg

    return 1, 0, msg


def handle_user_connection(user_register_dict):

    def wrapper(func):
        @wraps(func)
        async def func_wrapper(*args, **kwargs):
            web_socket_inst = None
            for _ in args:
                if not isinstance(_, WebSocketServerProtocol):
                    continue
                web_socket_inst = _
                break

            if web_socket_inst is None:
                for _ in kwargs.values():
                    if not isinstance(_, WebSocketServerProtocol):
                        continue
                    web_socket_inst = _
                    break

            if not isinstance(web_socket_inst, WebSocketServerProtocol):
                raise ValueError('不支持的函数类型')

            headers = web_socket_inst.request_headers
            if 'user_info' not in headers:
                raise ConnectionRefusedError('user not login')

            user_info = headers['user_info']
            if 'company_user_id' not in user_info:
                raise ConnectionRefusedError('user id not found')

            company_user_id = user_info['company_user_id']

            if company_user_id in user_register_dict:
                try:
                    await user_register_dict[company_user_id].close()
                except:
                    pass

            user_register_dict[company_user_id] = web_socket_inst
            try:
                rtn_result = await func(*args, **kwargs)
            except:
                rtn_result = None
            finally:
                user_register_dict.pop(company_user_id, None)

            return rtn_result

        return func_wrapper

    return wrapper




