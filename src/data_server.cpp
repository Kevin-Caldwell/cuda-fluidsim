#include "data_server.h"
#include <assert.h>
#include <cstdio>
#include <sys/types.h>
#include <unistd.h>

char ack_msg[] = "ACK";
cmd_t cmd_buf = -1;

data_server::data_server(const char port_name[MAX_STR_LEN], int size_x,
                         int size_y) {
  int res = 0;
  this->size_x = size_x;
  this->size_y = size_y;
  this->byte_size = size_x * size_y * sizeof(float);

  this->server_addr.sun_family = AF_UNIX;
  snprintf(this->server_addr.sun_path, MAX_STR_LEN, "%s.sock", port_name);

  // Initialize Server File Descriptor
  this->server_fd = socket(AF_UNIX, SOCK_STREAM, AUTOMATIC_PROTOCOL);
  if (this->server_fd == -1) {
    perror("ERROR:");
  } else {
    printf("SUCCESS: Socket Created\n");
  }

  // Bind Socket to an Address
  printf("Binding to address: %s\n", this->server_addr.sun_path);

  res = bind(this->server_fd, (struct sockaddr *)&(this->server_addr),
             sizeof(this->server_addr));
  if (res == -1) {
    perror(NULL);
  } else {
    printf("SUCCESS: Bind to Socket\n");
  }

  res = listen(this->server_fd, 3);
  if (res == -1) {
    printf("Unable to Connect\n");
    perror(NULL);
  } else {
    printf("Socket set to Listen\n");
  }

  // Establish Connection Socket
  snprintf(this->connection_addr.sun_path, MAX_STR_LEN, "active_%s.sock",
           port_name);

  this->connection_addr.sun_family = AF_UNIX;
  socklen_t connected_addr_len = 0;

  int frame_sz[2] = {this->size_x, this->size_y};

  this->connection_fd =
      accept(this->server_fd, (struct sockaddr *)&this->connection_addr,
             &connected_addr_len);

  printf("Successfully Connected to %s\n", this->connection_addr.sun_path);

  res = write(this->connection_fd, &frame_sz, sizeof(frame_sz));
  printf("Written %d Bytes\n", res);
}

data_server::~data_server() {
  printf("Destroying Server...\n");
  if (this->server_fd) {
    cmd_buf = STOP;
    ssize_t write_sz = write(this->server_fd, &cmd_buf, sizeof(cmd_t));
    close(this->server_fd);
    remove(this->server_addr.sun_path);
  }
  if (this->connection_fd) {
    cmd_buf = STOP;
    ssize_t write_sz = write(this->connection_fd, &cmd_buf, sizeof(cmd_t));

    close(this->connection_fd);
    remove(this->connection_addr.sun_path);
  }
}

int data_server::send_frame(float *src) {
  cmd_buf = SENDING;
  ssize_t bytes_written = write(this->connection_fd, &cmd_buf, sizeof(cmd_t));
  if (bytes_written == sizeof(cmd_t)) {
    bytes_written = 0;
    while (bytes_written < this->byte_size) {

      bytes_written = write(this->connection_fd, src, this->byte_size);
      // printf("Written %ld bytes\n", (long int) bytes_written);
    }
  } else {
    // printf("Failed to Send Write Command.");
  }

  bytes_written = read(this->connection_fd, &cmd_buf, sizeof(cmd_t));
  if (cmd_buf != ACK) {
    // printf("Client did not receive\n");
  }

  return 0;
}
