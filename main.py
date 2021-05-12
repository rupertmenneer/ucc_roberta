import flask
import json
import options_schema
from ucc_classifier_pkg import ucc_classifer_wrapper
from sensible_result import make_sensible

VERBOSE = 1


def create_app():

    application = flask.Flask(__name__)
    # Load the model
    ucc_model = ucc_classifer_wrapper.ClassifierWrapper()
    schema = options_schema.schema
    print('INFO: model server ready')

    @application.route("/options", methods=["GET"])
    def options():
        return flask.jsonify(schema)

    @application.route("/predict", methods=["POST"])
    def predict():
        # Get JSON payload
        m52_input = flask.request.get_json()
        if VERBOSE:
            print('RECEIVED:', json.dumps(m52_input, indent=2))
        # Transform to DataFrame usable by classifier
        comment_list = [c['comment'] for c in m52_input]
        print(comment_list)
        result = ucc_model.classify_raw_comments(comment_list)
        sensible_result = make_sensible(result)
        if VERBOSE:
            print('SENDING:', json.dumps(sensible_result, indent=2))
        # Build and send response
        resp = flask.Response(
            response=json.dumps(sensible_result),
            status=200,
            mimetype="application/json"
        )
        return resp

    return application


# if __name__ == '__main__':
#     application.run(debug=True, use_reloader=False)
