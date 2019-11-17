from flask import Flask, send_file, make_response, render_template
import os
from datetime import datetime
from threading import Lock

mutex = Lock()

def start_app(env):

    app = Flask(__name__)

    BASE_URL = "http://localhost:5000/"

    @app.route('/')
    def index():
        return render_template('index.html', base_url=BASE_URL)

    @app.route('/plots/q-table', methods=['GET'])
    def q_table():  
        mutex.acquire()
        bytes_obj = env.get_qtable_png()
        mutex.release()

        if bytes_obj is not None:
            return send_file(bytes_obj,
                            attachment_filename=str(datetime.now()).split('.')[0] + '_qtable.png',
                            mimetype='image/png')
        else:
            return make_response('Not enough data to plot this.')


    @app.route('/plots/rewards', methods=['GET'])
    def rewards():
        mutex.acquire()
        bytes_obj = env.get_rewards()
        mutex.release()

        if bytes_obj is not None:
            return send_file(bytes_obj,
                            attachment_filename=str(datetime.now()).split('.')[0] + '_rewards.png',
                            mimetype='image/png')
        else:
            return make_response('Not enough data to plot this.')

    @app.route('/plots/episoderewards', methods=['GET'])
    def episoderewards():
        mutex.acquire()
        bytes_obj = env.get_episoderewards()
        mutex.release()

        if bytes_obj is not None:
            return send_file(bytes_obj,
                            attachment_filename=str(datetime.now()).split('.')[0] + '_episoderewards.png',
                            mimetype='image/png')
        else:
            return make_response('Not enough data to plot this.')

    @app.route('/plots/featuresdistribution', methods=['GET'])
    def featuresdistribution():
        mutex.acquire()
        bytes_obj = env.get_featuresdistribution()
        mutex.release()

        if bytes_obj is not None:
            return send_file(bytes_obj,
                            attachment_filename=str(datetime.now()).split('.')[0] + '_featuresdistribution.png',
                            mimetype='image/png')
        else:
            return make_response('Not enough data to plot this.')

    @app.route('/plots/actionsdistribution', methods=['GET'])
    def actionsdistribution():
        mutex.acquire()
        bytes_obj = env.get_actionsdistribution()
        mutex.release()

        if bytes_obj is not None:
            return send_file(bytes_obj,
                            attachment_filename=str(datetime.now()).split('.')[0] + '_actionsdistribution.png',
                            mimetype='image/png')
        else:
            return make_response('Not enough data to plot this.')

    app.jinja_env.auto_reload = True
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=False)