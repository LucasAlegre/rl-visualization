from flask import Flask, send_file, make_response, render_template, request
import os
from datetime import datetime
from threading import Lock

mutex = Lock()

class Plot:
    def __init__(self, title, url):
        self.title = title
        self.url = url

def start_app(env):

    app = Flask(__name__)

    BASE_URL = "http://localhost:5000/"

    plots_url = {
        'Q-table': 'plots/q-table',
        'Visit Count': 'plots/visitcount',
        'Rewards': 'plots/rewards',
        'Episode Rewards': 'plots/episoderewards',
        'Epsilon': 'plots/epsilon',
        'Features Distributions': 'plots/featuresdistribution',
        'Actions Distributions': 'plots/actionsdistribution'
    }

    @app.route('/')
    def index():
        delay = request.args.get('delay')
        if delay is not None:
            env.delay = int(delay)

        plots = env.get_available_plots()
        return render_template('index.html', base_url=BASE_URL, plots=[Plot(p, plots_url[p]) for p in plots], refresh_time=env.refresh_time)

    @app.route('/plots/visitcount', methods=['GET'])
    def visitcount():
        mutex.acquire()
        bytes_obj = env.get_visitcount()
        mutex.release()

        return send_file(bytes_obj, attachment_filename='visitcount.png', mimetype='image/png')     

    @app.route('/plots/q-table', methods=['GET'])
    def q_table():  
        mutex.acquire()
        bytes_obj = env.get_qtable_png()
        mutex.release()

        return send_file(bytes_obj, attachment_filename='qtable.png', mimetype='image/png')

    @app.route('/plots/rewards', methods=['GET'])
    def rewards():
        mutex.acquire()
        bytes_obj = env.get_rewards()
        mutex.release()

        return send_file(bytes_obj, attachment_filename='rewards.png', mimetype='image/png')

    @app.route('/plots/episoderewards', methods=['GET'])
    def episoderewards():
        mutex.acquire()
        bytes_obj = env.get_episoderewards()
        mutex.release()

        return send_file(bytes_obj, attachment_filename='episoderewards.png', mimetype='image/png')

    @app.route('/plots/epsilon', methods=['GET'])
    def epsilon():
        mutex.acquire()
        bytes_obj = env.get_epsilon()
        mutex.release()

        return send_file(bytes_obj, attachment_filename='epsilon.png', mimetype='image/png')

    @app.route('/plots/featuresdistribution', methods=['GET'])
    def featuresdistribution():
        mutex.acquire()
        bytes_obj = env.get_featuresdistribution()
        mutex.release()

        return send_file(bytes_obj, attachment_filename='featuresdistribution.png', mimetype='image/png')

    @app.route('/plots/actionsdistribution', methods=['GET'])
    def actionsdistribution():
        mutex.acquire()
        bytes_obj = env.get_actionsdistribution()
        mutex.release()

        return send_file(bytes_obj, attachment_filename='actionsdistribution.png', mimetype='image/png')

    app.jinja_env.auto_reload = True
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=False)