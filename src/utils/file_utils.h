#ifndef CVECTOR_FILE_UTILS_H
#define CVECTOR_FILE_UTILS_H

#include <stdint.h>
#include <stddef.h>

// File utility functions for CVector
int cvector_ensure_directory(const char* path);
int cvector_file_exists(const char* path);
size_t cvector_file_size(const char* path);
int cvector_create_backup(const char* original_path, const char* backup_path);

#endif // CVECTOR_FILE_UTILS_H
