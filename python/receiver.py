import socket
import struct
import sys
import numpy as np
import matplotlib.pyplot as plt
import time

def bytearray_to_double_array(bytearray):
    """Converts a bytearray to an array of 64-bit doubles.

    Args:
        bytearray: The input bytearray.

    Returns:
        A list of 64-bit doubles.
    """

    if len(bytearray) % 8 != 0:
        raise ValueError("Bytearray length must be a multiple of 8.")

    num_doubles = len(bytearray) // 4
    fmt = '<' + 'f' * num_doubles  # '<' for little-endian, 'd' for double
    return struct.unpack(fmt, bytearray)


if __name__ == "__main__":
    
    socket_fd = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM, 0)

    connection = 0
    trials = 0
    connected = False

    while not connected:
        try:
            connection = socket_fd.connect(f"{sys.argv[1]}.sock")
            connected = True
        except:
            connected = False
            time.sleep(0.1)
            trials += 1
            if trials > 60:
                break
            # print("Trying Again")
            

    data_in = socket_fd.recv(8)
    frame_sz = struct.unpack('<ii', data_in)
    frame_val = frame_sz[0] * frame_sz[1]

    i = 0
    print(f"Frame Size: {frame_sz}")


    while True:
        data_in = socket_fd.recv(4)
        cmd_recv = struct.unpack('<i', data_in)[0]

        match cmd_recv:
            case 1:
               socket_fd.send("ACK")



            case 2: # Sending
                buffer_count = 0
                data = []
                a = 1
                while buffer_count < frame_val:
                    # print("Attempting to read Values")

                    data_in = socket_fd.recv(frame_sz[0] * frame_sz[1] * 8)
                    formatted_data = bytearray_to_double_array(data_in)

                    data.extend(formatted_data)
                    buffer_count += len(formatted_data)

                    # print(f"Received {len(data)}/{frame_val} values")

                # print("SENT ACK")
                socket_fd.send(bytes(a))
                
                frame = np.reshape(data, frame_sz)

                plt.cla()
                plt.imshow(frame.transpose(), cmap='gray')
                plt.title(f"({np.min(frame)}, {np.max(frame)})")
                plt.savefig(f"temp/{sys.argv[1]}"+"%02d.png" % i)

                i += 1


            case 3: # Recving
                continue



            case 4: # Stop
                break
