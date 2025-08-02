#pragma once

#include <limits>

#include "errors.h"
#include "fsim/sim_params.h"

namespace backup
{

constexpr int max_file_length = 255;

class Backup
{
 public:
  Backup(const bool reset_count, SimParams *params);
  ~Backup();

 private:
  const char* metadata_location = "data/backups/.metadata";
  const char* backup_folder_location = "data/backups";
  const char* return_folder = "../../../";

  bool reset_count_;
  char backup_location_[max_file_length];
  int backup_count_;
  SimParams *params_;
};

ret_t setup_backup();

ret_t exit_backup(SimParams *p);

}  // namespace backup
