
#include "file_utils.h"
#include <sys/stat.h>
#include <string.h>
#include <unistd.h>
#include <libgen.h>
#include <stdlib.h>
#include <stdio.h>

int cvector_ensure_directory(const char* path) {
    if (!path) return 0;
    
    char* path_copy = strdup(path);
    if (!path_copy) return 0;
    
    char* dir = dirname(path_copy);
    
    struct stat st;
    int result = 1;
    
    if (stat(dir, &st) != 0) {
        // Directory doesn't exist, try to create it
        if (mkdir(dir, 0755) != 0) {
            result = 0;
        }
    } else if (!S_ISDIR(st.st_mode)) {
        // Path exists but is not a directory
        result = 0;
    }
    
    free(path_copy);
    return result;
}

int cvector_file_exists(const char* path) {
    if (!path) return 0;
    
    struct stat st;
    return (stat(path, &st) == 0);
}

size_t cvector_file_size(const char* path) {
    if (!path) return 0;
    
    struct stat st;
    if (stat(path, &st) != 0) {
        return 0;
    }
    
    return (size_t)st.st_size;
}

int cvector_create_backup(const char* original_path, const char* backup_path) {
    if (!original_path || !backup_path) return 0;
    
    FILE* src = fopen(original_path, "rb");
    if (!src) return 0;
    
    FILE* dst = fopen(backup_path, "wb");
    if (!dst) {
        fclose(src);
        return 0;
    }
    
    char buffer[8192];
    size_t bytes;
    int success = 1;
    
    while ((bytes = fread(buffer, 1, sizeof(buffer), src)) > 0) {
        if (fwrite(buffer, 1, bytes, dst) != bytes) {
            success = 0;
            break;
        }
    }
    
    fclose(src);
    fclose(dst);
    
    if (!success) {
        unlink(backup_path);
    }
    
    return success;
}