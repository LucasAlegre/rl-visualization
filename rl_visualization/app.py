from flask import Flask, send_file, make_response, render_template
import os
from datetime import datetime


def start_app(env):

    app = Flask(__name__)

    BASE_URL = "http://localhost:5000/"

    @app.route('/')
    def index():
        return render_template('index.html', base_url=BASE_URL)

    @app.route('/plots/featuresdistribution', methods=['GET'])
    def featuresdistribution():
        bytes_obj = env.get_featuresdistribution()

        return send_file(bytes_obj,
                        attachment_filename=str(datetime.now()).split('.')[0] + '_featuresdistribution.png',
                        mimetype='image/png')


    @app.route('/plots/q-table', methods=['GET'])
    def q_table():
        try:
            bytes_obj = env.get_qtable_png()

            return send_file(bytes_obj,
                            attachment_filename=str(datetime.now()).split('.')[0] + '_qtable.png',
                            mimetype='image/png')
        except ValueError:
            return make_response('Unsupported request, probably feature names are wrong', 400)


    app.jinja_env.auto_reload = True
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=False)