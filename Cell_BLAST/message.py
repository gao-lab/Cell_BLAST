"""
Utility functions for printing messages
"""


def message(msg, header):
    print("[{0:^9}] {1}".format(header, msg))


def info(msg):
    message(msg, "Info")


def warning(msg):
    message(msg, "Warning")
