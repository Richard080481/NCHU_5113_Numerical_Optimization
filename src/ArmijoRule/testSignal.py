import signal
import sys
import time

def signal_handler(sig, frame):
    print(f"receive {sig}")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGABRT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
print('Press Ctrl+C')
time.sleep(20)