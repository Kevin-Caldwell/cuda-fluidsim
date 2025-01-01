#ifndef DATA_SERVER_H
#define DATA_SERVER_H

#include <sys/socket.h>
#include <iostream>
#include <sys/un.h>

#define MAX_STR_LEN 108
#define AUTOMATIC_PROTOCOL (0)

typedef unsigned int cmd_t;

enum CMD_LIST: unsigned int{
    NO_ACTION = 0, 
    ACK, 
    SENDING,
    RECVING, 
    STOP,
};

/**
 * @class data_server
 * @brief Transmits Frames from FluidSim to Python Receiver 
 */
class data_server
{
private:
    // File Descriptor for handling IO to/from Socket
    int server_fd, connection_fd;
    int size_x, size_y;
    int byte_size;
    struct sockaddr_un server_addr;
    struct sockaddr_un connection_addr;

public:

    data_server(
        const char port_name[MAX_STR_LEN], 
        int size_x, 
        int size_y
    );

    ~data_server();

    int send_frame(float* src);
};



#endif /* DATA_SERVER_H */
