import fibsem

# AutoLamella is not packaged separately — it ships inside the `fibsem`
# distribution, so importlib.metadata.version('autolamella') raises
# PackageNotFoundError. Its version is therefore fibsem's, which is what the
# About dialog (fibsem/ui/utils.py) has always reported for it.
__version__ = fibsem.__version__
