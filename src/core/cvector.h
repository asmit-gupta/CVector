#ifndef CVECTOR_H
#define CVECTOR_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

// Version info
#define CVECTOR_VERSION_MAJOR 0
#define CVECTOR_VERSION_MINOR 1
#define CVECTOR_VERSION_PATCH 0

// Constants
#define CVECTOR_MAX_DIMENSION 4096
#define CVECTOR_DEFAULT_DIMENSION 512
#define CVECTOR_MAX_DB_NAME 256
#define CVECTOR_MAX_PATH 1024

// Error codes
typedef enum {
    CVECTOR_SUCCESS = 0,
    CVECTOR_ERROR_INVALID_ARGS = -1,
    CVECTOR_ERROR_OUT_OF_MEMORY = -2,
    CVECTOR_ERROR_FILE_IO = -3,
    CVECTOR_ERROR_DB_NOT_FOUND = -4,
    CVECTOR_ERROR_VECTOR_NOT_FOUND = -5,
    CVECTOR_ERROR_DIMENSION_MISMATCH = -6,
    CVECTOR_ERROR_DB_CORRUPT = -7
} cvector_error_t;

// Similarity metrics
typedef enum {
    CVECTOR_SIMILARITY_COSINE = 0,
    CVECTOR_SIMILARITY_DOT_PRODUCT = 1,
    CVECTOR_SIMILARITY_EUCLIDEAN = 2
} cvector_similarity_t;

// Vector ID type
typedef uint64_t cvector_id_t;

// Vector structure
typedef struct {
    cvector_id_t id;
    uint32_t dimension;
    float* data;
    uint64_t timestamp;  // Creation/update timestamp
} cvector_t;

// Database configuration
typedef struct {
    char name[CVECTOR_MAX_DB_NAME];
    char data_path[CVECTOR_MAX_PATH];
    uint32_t dimension;
    cvector_similarity_t default_similarity;
    bool memory_mapped;
    size_t max_vectors;
} cvector_db_config_t;

// Database handle
typedef struct cvector_db cvector_db_t;

// Query result
typedef struct {
    cvector_id_t id;
    float similarity;
    cvector_t* vector;  // Optional: full vector data
} cvector_result_t;

// Query structure
typedef struct {
    float* query_vector;
    uint32_t dimension;
    uint32_t top_k;
    cvector_similarity_t similarity;
    float min_similarity;  // Filter threshold
} cvector_query_t;

// Core Database Operations
cvector_error_t cvector_db_create(const cvector_db_config_t* config, cvector_db_t** db);
cvector_error_t cvector_db_open(const char* db_path, cvector_db_t** db);
cvector_error_t cvector_db_close(cvector_db_t* db);
cvector_error_t cvector_db_drop(const char* db_path);

// Vector CRUD Operations
cvector_error_t cvector_insert(cvector_db_t* db, const cvector_t* vector);
cvector_error_t cvector_insert_batch(cvector_db_t* db, const cvector_t* vectors, size_t count);
cvector_error_t cvector_get(cvector_db_t* db, cvector_id_t id, cvector_t** vector);
cvector_error_t cvector_update(cvector_db_t* db, const cvector_t* vector);
cvector_error_t cvector_delete(cvector_db_t* db, cvector_id_t id);

// Query Operations
cvector_error_t cvector_search(cvector_db_t* db, const cvector_query_t* query, 
                              cvector_result_t** results, size_t* result_count);

// Utility Functions
cvector_error_t cvector_create_vector(cvector_id_t id, uint32_t dimension, 
                                     const float* data, cvector_t** vector);
void cvector_free_vector(cvector_t* vector);
void cvector_free_results(cvector_result_t* results, size_t count);
const char* cvector_error_string(cvector_error_t error);

// Database Stats
typedef struct {
    size_t total_vectors;
    size_t total_size_bytes;
    uint32_t dimension;
    cvector_similarity_t default_similarity;
    char db_path[CVECTOR_MAX_PATH];
} cvector_db_stats_t;

cvector_error_t cvector_db_stats(cvector_db_t* db, cvector_db_stats_t* stats);

#endif // CVECTOR_H