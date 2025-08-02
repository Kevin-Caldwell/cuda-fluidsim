#include "io/backups.h"

#include <sys/stat.h>
#include <unistd.h>

#include <string>

namespace backup
{

Backup::Backup(bool reset_count, SimParams *params)
    : reset_count_(reset_count), params_(params)
{
  // Open Metadata file
  int res = RES_OK;
  char cwd[100];

  getcwd(cwd, 100);
  printf("Current working drectory: %s\n", cwd);
  FILE *metadata_fp = fopen(metadata_location, "r+");
  if (metadata_fp == NULL) {
    perror("Backups Folder not setup correctly\n");
    exit(BACKUP_INIT_FAILED);
  }

  // Read Backup Count
  if (!reset_count) {
    res = fread(&backup_count_, sizeof(backup_count_), 1, metadata_fp);
    if (res == 0) {
      backup_count_ = 0;
    }

    backup_count_ += 1;
  } else {
    backup_count_ = 0;
  }

  printf("%d Backup Count\n", backup_count_);

  // Write Backup count
  fseek(metadata_fp, 0, 0);
  fwrite(&backup_count_, sizeof(backup_count_), 1, metadata_fp);
  fclose(metadata_fp);

  snprintf(backup_location_,
           max_file_length,
           "%s/b%05d",
           backup_folder_location,
           backup_count_);
  mkdir(backup_location_, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

  // Move into backup location
  chdir(backup_location_);
  printf("Backup Created.\n");
}

Backup::~Backup()
{
  write_parameters_to_file("backup.config", params_);
  chdir(return_folder);
}

}  // namespace backup
