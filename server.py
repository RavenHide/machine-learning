# -*- coding: utf-8 -*-
"""
    server
    ~~~~~~~

    Description.

    :author: zhaoyouqing
    :copyright: (c) 2018, Tungee
    :date created: 2020-06-15
    :python version: 3.5
"""


from gevent import monkey
# Monkey patch to support coroutine
# Must be at the start of the whole app


monkey.patch_all()

try:
    from gevent.wsgi import WSGIServer
except Exception:
    from gevent.pywsgi import WSGIServer

from auth import login_auth
from flask import Flask, Blueprint, jsonify, current_app, request
from flask_sockets import Sockets
from geventwebsocket.handler import WebSocketHandler

api = Blueprint('api', __name__)
ws_api = Blueprint('/ws-api', __name__)


@api.route('/hello')
def hello():
    print(current_app.extensions)
    user_socket_dict = current_app.extensions['user_socket_dict']
    company_user_id = '5be0fab369975972b878b05b'
    socket_inst = user_socket_dict[company_user_id]
    if not socket_inst.closed:
        socket_inst.send('feedback')
    return jsonify(stat=1, msg='hello world'), 200


@ws_api.route('/echo')
@login_auth
def echo_socket(socket, **kwargs):
    # print(request.cookies.get('accountCenterSessionId'))
    print(kwargs)
    while not socket.closed:
        message = socket.receive()
        socket.send(message)


@ws_api.route('/echo2')
def echo_socket(socket):
    while not socket.closed:
        print(socket.handler.server.clients)
        # message = socket.receive()
        for k, v in socket.handler.server.clients.items():
            v.ws.send('feed back')

        socket.closed()
        # socket.send(message + a)



app = Flask(__name__)

app.secret_key = '000000'
sockets = Sockets(app)

app.register_blueprint(api, url_prefix='/api')
sockets.register_blueprint(ws_api, url_prefix='/ws-api')

app.extensions['user_socket_dict'] = dict()

if __name__ == '__main__':
    from werkzeug.serving import run_with_reloader
    server = WSGIServer(('localhost', 10202), app, handler_class=WebSocketHandler)

    @run_with_reloader
    def run_server():
        server.serve_forever()

    # server.serve_forever()
