from threading import Thread
import queue
import time
import sys

#source by Shashwat Kumar https://medium.com/@shashwat_ds/a-tiny-multi-threaded-job-queue-in-30-lines-of-python-a344c3f3f7f0
class TaskQueue(queue.Queue):

    def __init__(self, num_workers=1):
        queue.Queue.__init__(self)
        self.num_workers = num_workers
        self.start_workers()

    def add_task(self, task, *args, **kwargs):
        args = args or ()
        kwargs = kwargs or {}
        self.put((task, args, kwargs))
        print("added", args)

    def start_workers(self):
        for i in range(self.num_workers):
            t = Thread(target=self.worker)
            t.daemon = True
            t.start()
            print('thread started: ', i)

    def worker(self):
        while True:
            try:
                item, args, kwargs = self.get()
                item(*args, **kwargs)
                self.task_done()
            except:
                print("Unexpected error:", sys.exc_info()[0])

def tests():
    def blokkah(*args, **kwargs):
        print(args[0])

    q = TaskQueue(num_workers=1)

    for item in range(10):
        q.add_task(blokkah, item)

    q.join()       # block until all tasks are done
    print("All done!")

tests()