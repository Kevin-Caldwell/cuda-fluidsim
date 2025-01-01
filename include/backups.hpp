#include <string>
#include <sys/stat.h>
#include <unistd.h>

#include "errors.hpp"
#include "sim_params.hpp"

const unsigned int max_file_length = 100;

const char metadata_location[] = "data/backups/.metadata";
const char backup_folder_location[] = "data/backups";
const char return_folder[] = "../../../";
const bool reset_count = false;

extern char backup_location[];

extern int backup_count;

ret_t setup_backup();

ret_t exit_backup(SimParams* p);

