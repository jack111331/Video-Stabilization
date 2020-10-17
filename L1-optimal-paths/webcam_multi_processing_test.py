import cv2
from multiprocessing import Process, Pipe

cap = cv2.VideoCapture(0)


def dummy():
    return


def launch_read_frames(connection_obj):
    """
    Starts the read_frames function in a separate process.
    param connection_obj: pipe to send frames into.
    Returns a pipe object
    """
    # Create a new process in the stype of C threads where a particular role is assigned to it
    read_frames_process = Process(target=dummy, args=(connection_obj,))
    # Start the new process
    read_frames_process.start()
    # Wait for the new process to terminate by blocking till we reach it
    read_frames_process.join()

    return parent_conn


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


if __name__ == '__main__':
    # Create a pipe with two ends a parent and a child
    # Pipes are used to communicate between 2 processes
    parent_conn, child_conn = Pipe()
    # Call a function that creates a new child process and
    # links it to the 'output'/child end of the pipe
    launch_read_frames(child_conn)

    # You could also call this function as a separate process, though in
    # this instance I see no performance benefit.
    act_on_frames(parent_conn)