import cv2
import multiprocessing as mp
from multiprocessing import Process, Pipe

cap = cv2.VideoCapture(0)


# Worker function in worker-producer setup
def worker(i):
    print("Namaste, mainh wroker number {0} hun, mera nam {1} hain".format(i, mp.current_process().name))
    return


def launch_read_frames(connection_obj):
    """
    Starts the worker function in a separate process.
    param connection_obj: pipe to send frames into.
    Returns a pipe object
    """
    # Create a new process in the stype of C threads where a particular role is assigned to it
    read_frames_process = Process(target=worker, args=(connection_obj,))
    # Start the new process
    read_frames_process.start()
    # Wait for the new process to terminate by blocking till we reach it
    read_frames_process.join()
    return


if __name__ == '__main__':
    jobs = []
    for i in range(5):
        p = Process(name='Worker-{0}'.format(i), target=worker, args=(i, ))
        jobs.append(p)
        p.start()
        # p.join()
        print("Does this print first {0} from process {1}".format(i, mp.current_process().name))
        # p.join()
