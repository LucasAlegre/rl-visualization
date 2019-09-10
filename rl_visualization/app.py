from flask import Flask, send_file, make_response, render_template
import os
from datetime import datetime
from rl_visualization.plotting import get_correlation_matrix_as_bytes, get_breast_cancer_df, get_pair_plot_as_bytes


def start_app(env):

    app = Flask(__name__)

    # get data to keep it in memory, usually you will serve this from a database or bucket, or whereever your db is sitting
    breast_cancer_df, features_names = get_breast_cancer_df()

    BASE_URL = "http://localhost:5000/"

    @app.route('/')
    def index():
        return render_template('index.html', base_url=BASE_URL)

    @app.route('/plots/breast_cancer_data/pairplot/features/<features>', methods=['GET'])
    def pairplot(features):
        try:
            # parse columns
            parsed_features = [feature.strip() for feature in features.split(',')]
            bytes_obj = get_pair_plot_as_bytes(breast_cancer_df, parsed_features)

            return send_file(bytes_obj,
                            attachment_filename='plot.png',
                            mimetype='image/png')
        except ValueError:
            # something went wrong to return bad request
            return make_response('Unsupported request, probably feature names are wrong', 400)


    @app.route('/plots/q-table', methods=['GET'])
    def correlation_matrix():
        bytes_obj = env.get_qtable_png()

        return send_file(bytes_obj,
                        attachment_filename=str(datetime.now()).split('.')[0] + '_plot.png',
                        mimetype='image/png')

    app.jinja_env.auto_reload = True
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=False)