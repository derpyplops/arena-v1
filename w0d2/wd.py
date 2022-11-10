import sys
import logging
from watchdog.observers import Observer
from watchdog.events import LoggingEventHandler

class Event(LoggingEventHandler):
    def __init__(self, logger=None):
        super().__init__(logger)
        self.counter = 0

    def on_modified(self, event):
        print(f"===== {self.counter} =====")
        self.counter += 1
        print(event.src_path)
        try:
            exec(open(event.src_path).read())
        except Exception as e:
            print(e)

            

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    path = sys.argv[1] if len(sys.argv) > 1 else '.'
    event_handler = Event()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()
    try:
        while observer.is_alive():
            observer.join(1)
    finally:
        observer.stop()
        observer.join()
