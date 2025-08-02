#include "io/backups.h"

#include <sys/stat.h>
#include <unistd.h>

#include <string>

namespace backup
{

char backup_location[max_file_length];
int backup_count = 0;

ret_t setup_backup()
{
  /** OPEN METADATA FILE */
  int res = RES_OK;
  FILE *metadata_fp = fopen(metadata_location, "r+");
  if (metadata_fp == NULL) {
    printf("Backups Folder not setup correctly\n");
    return ERR_FOPEN;
  }

  /** READ BACKUP COUNT */
  if (!reset_count) {

    res = fread(&backup_count, sizeof(backup_count), 1, metadata_fp);
    if (res == 0) {
      backup_count = 0;
    }

    backup_count += 1;
  } else {
    backup_count = 0;
  }

  /** WRITE NEW BACKUP COUNT */
  fseek(metadata_fp, 0, 0);
  fwrite(&backup_count, sizeof(backup_count), 1, metadata_fp);

  snprintf(backup_location, max_file_length, "%s/b%05d", backup_folder_location,
           backup_count);
  mkdir(backup_location, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

  fclose(metadata_fp);

  /** MOVE INTO BACKUP LOCATION */
  chdir(backup_location);

  return RES_OK;
}

ret_t exit_backup(SimParams *p)
{
  write_parameters_to_file("backup.config", p);
  chdir(return_folder);
  return RES_OK;
}

Backup::Backup(bool reset_count, SimParams *params)
    : reset_count_(reset_count), params_(params)
{
  // Open Metadata file
  int res = RES_OK;
  char cwd[100];

  getcwd(cwd, 100);
  printf("Current working drectory: %s", cwd);
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

  // Write Backup count
  fseek(metadata_fp, 0, 0);
  fwrite(&backup_count_, sizeof(backup_count_), 1, metadata_fp);

  snprintf(backup_location_, max_file_length, "%s/b%05d",
           backup_folder_location, backup_count);
  mkdir(backup_location_, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

  fclose(metadata_fp);

  // Move into backup location
  chdir(backup_location);
}

Backup::~Backup()
{
  write_parameters_to_file("backup.config", params_);
  chdir(return_folder);
}

}  // namespace backup
