import os
import webbrowser
import threading
import time

from flask import Flask, jsonify, render_template, request
from flask_socketio import SocketIO
import numpy as np

from ..service_base import BaseFlaskService


_FLASK_TEMPLATE_DIR = os.path.dirname(os.path.abspath(__file__))
_FLASK_TEMPLATE_DIR = os.path.join(_FLASK_TEMPLATE_DIR, 'templates')

app = Flask(__name__, template_folder=_FLASK_TEMPLATE_DIR)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

# Generate a random point cloud
def generate_point_cloud(N):
    return np.random.rand(N, 3) * 1  # N points in 3D space

@app.route('/')
def index():
    return render_template('index.html')  # Serve the Three.js frontend page

@app.route('/getServerUrl')
def get_server_url():
    return request.base_url

@app.route('/getPointCloud')
def get_point_cloud():
    N = 10  # Number of points
    point_cloud = generate_point_cloud(N)
    return jsonify(point_cloud.tolist())  # Convert numpy array to list for JSON serialization

@socketio.on('connect')
def test_connect():
    print('Client connected')

@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected')


class Service(BaseFlaskService):
    def __init__(
        self
    ) -> None:
        self.app=app
        self.socketio=socketio
        
    def update_graph(self, graph):
        self.socketio.emit('update_graph', graph.jsonify())