import os
import webbrowser
import threading
import time
import typing
from abc import ABCMeta, abstractmethod

from flask import Flask, jsonify, render_template, request
from flask_socketio import SocketIO
import numpy as np


class BaseFlaskService(metaclass=ABCMeta):
    app: Flask
    socketio: SocketIO
    host: str = '127.0.0.1'
    port: int = 5000
    _app_thread: typing.Optional[threading.Thread] = None
    
    @abstractmethod
    def __init__(self):
        _app_thread = None
    
    def open_browser(self):
        webbrowser.open_new_tab(f'http://{self.host}:{self.port}/')
        time.sleep(2.0)

    def run_app(
        self,
        host: str | None = None,
        port: int | None = None,
        debug: bool | None = None,
        load_dotenv: bool = True,
        **kwargs: typing.Any,
    ) -> None:
        """
            If you experience any problem with Jupyter, set 'use_reloader=False' to True.
        """
        self.host = host if host is not None else self.host
        self.port = port if port is not None else self.port
        if "use_reloader" not in kwargs.keys():
            kwargs["use_reloader"] = False         # To avoid collision with Jupyter.

        def _run():
            self.app.run(host=self.host, port=self.port, debug=debug, load_dotenv=load_dotenv, **kwargs)
            
        # threading.Timer(1.25, lambda: self.open_browser()).start()
        self._app_thread = threading.Thread(target = _run, daemon=True)
        self._app_thread.start()
        time.sleep(0.3)
        self.open_browser()
        
    def open_app(self, *args, **kwargs):
        if self._app_thread is None:
            self.run_app(*args, **kwargs)
        elif not self._app_thread.is_alive():
            self.run_app(*args, **kwargs)
        else:
            self.open_browser()









        
    
    