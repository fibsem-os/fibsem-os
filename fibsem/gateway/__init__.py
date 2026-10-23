"""The gateway: an always-on process on a microscope PC that runs one mode at a
time (nothing, the app's window, or the app with no window), so a hub on
another computer can start and stop apps there. It never touches the
microscope itself; the app it starts does."""
