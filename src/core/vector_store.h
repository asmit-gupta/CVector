#ifndef CVECTOR_VECTOR_STORE_H
#define CVECTOR_VECTOR_STORE_H

#include "cvector.h"
#include <stdio.h>

// Internal structures and functions for vector storage
// This header is not exposed to external users

// Hash table entry for vector lookup
typedef struct cvector_vector_entry {
    cvector_id_t id;
    uint64_t file_offset;
    uint32_t dimension;
    uint64_t timestamp;
    bool is_deleted;
    struct cvector_vector_entry* next;
} cvector_vector_entry_t;

// Internal database structure declaration
struct cvector_db;

// Internal helper functions
static uint64_t cvector_hash(cvector_id_t id);
static uint64_t cvector_get_timestamp(void);
static cvector_error_t cvector_init_hash_table(cvector_db_t* db);
static void cvector_free_hash_table(cvector_db_t* db);
static cvector_error_t cvector_hash_insert(cvector_db_t* db, cvector_id_t id, 
                                          uint64_t file_offset, uint32_t dimension);
static cvector_vector_entry_t* cvector_hash_find(cvector_db_t* db, cvector_id_t id);
static cvector_error_t cvector_write_header(cvector_db_t* db);
static cvector_error_t cvector_read_header(cvector_db_t* db);

#endif // CVECTOR_VECTOR_STORE_H