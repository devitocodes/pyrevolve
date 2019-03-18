from flask_cors import CORS
from flask import Flask, request, jsonify, send_from_directory

from pyrevolve.schedulers import Revolve, Action


def static_schedule(nt, ncp):
    scheduler = Revolve(ncp, nt)
    schedule = []
    while True:
        next_action = scheduler.next()
        num_steps = scheduler.capo - scheduler.old_capo
        num_steps = max(1, abs(num_steps))

        for i in range(num_steps):
            old_action = None
            if next_action.type == Action.ADVANCE:
                old_action = next_action
                next_action = Action(next_action.type,
                                     next_action.oldcapo + i + 1,
                                     next_action.oldcapo + i,
                                     next_action.ckp)
            schedule.append(next_action)
            if old_action is not None:
                next_action = old_action
        if next_action.type == Action.TERMINATE:
            break

    return schedule


app = Flask(__name__)
CORS(app)


@app.route("/")
def index():
    return send_from_directory(".", "demo.html")


@app.route("/demo.js")
def script():
    return send_from_directory(".", "demo.js")


@app.route("/styles.css")
def style():
    return send_from_directory(".", "styles.css")


@app.route('/images/<path:filename>')
def image(filename):
    return send_from_directory("images", filename)


@app.route("/static_schedule")
def get_schedule():
    ncp = int(request.args.get('ncp'))
    nt = int(request.args.get('nt'))
    schedule = static_schedule(nt, ncp)
    schedule = [str(x) for x in schedule]
    return jsonify(schedule)
