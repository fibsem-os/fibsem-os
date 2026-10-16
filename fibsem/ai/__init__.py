"""Models for the app, scripts and agents: registered models, loaded once, run the
same way wherever they are called from.

``fibsem.ai.models`` holds the registry, the model cards and the manager; the
``inference`` service (`fibsem.services.inference`) serves them over the App
Server. This package imports nothing, so torch stays an optional install.
"""
