#include "cvector.h"
#include "vector_store.h"
#include "hnsw.h"
#include "similarity.h"
#include "../utils/file_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <unistd.h>
#include <pthread.h>

// Internal database structure
struct cvector_db {
    cvector_db_config_t config;
    FILE* data_file;
    FILE* index_file;
    FILE* metadata_file;
    cvector_id_t next_id;
    size_t vector_count;
    pthread_mutex_t mutex;          // Thread safety mutex
    pthread_rwlock_t search_lock;   // Read-write lock for searches
    bool is_open;
    
    // Simple hash table for vector lookup (in-memory for now)
    cvector_vector_entry_t** hash_table;
    size_t hash_table_size;
    
    // HNSW index for similarity search
    hnsw_index_t* hnsw_index;
};

// File format constants
#define CVECTOR_MAGIC_NUMBER 0x43564543  // "CVEC"
#define CVECTOR_FILE_VERSION 1
#define CVECTOR_BLOCK_SIZE 4096
#define CVECTOR_HASH_TABLE_SIZE 10007  // Prime number for good distribution

// File header structure
typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t dimension;
    cvector_similarity_t default_similarity;
    uint64_t vector_count;
    uint64_t next_id;
    uint64_t created_timestamp;
    uint64_t modified_timestamp;
    uint8_t reserved[32];  // For future use
} cvector_file_header_t;

// Vector file record structure
typedef struct {
    cvector_id_t id;
    uint32_t dimension;
    uint64_t timestamp;
    uint8_t is_deleted;
    uint8_t reserved[7];
    // Followed by dimension * sizeof(float) bytes of vector data
} cvector_vector_record_t;

// Helper functions
static uint64_t cvector_hash(cvector_id_t id) {
    return id % CVECTOR_HASH_TABLE_SIZE;
}

static uint64_t cvector_get_timestamp(void) {
    return (uint64_t)time(NULL);
}

static cvector_error_t cvector_init_hash_table(cvector_db_t* db) {
    db->hash_table_size = CVECTOR_HASH_TABLE_SIZE;
    db->hash_table = calloc(db->hash_table_size, sizeof(cvector_vector_entry_t*));
    return db->hash_table ? CVECTOR_SUCCESS : CVECTOR_ERROR_OUT_OF_MEMORY;
}

static void cvector_free_hash_table(cvector_db_t* db) {
    if (!db->hash_table) return;
    
    for (size_t i = 0; i < db->hash_table_size; i++) {
        cvector_vector_entry_t* entry = db->hash_table[i];
        while (entry) {
            cvector_vector_entry_t* next = entry->next;
            free(entry);
            entry = next;
        }
    }
    free(db->hash_table);
    db->hash_table = NULL;
}

static cvector_error_t cvector_hash_insert(cvector_db_t* db, cvector_id_t id, 
                                          uint64_t file_offset, uint32_t dimension) {
    uint64_t hash_idx = cvector_hash(id);
    cvector_vector_entry_t* entry = malloc(sizeof(cvector_vector_entry_t));
    if (!entry) return CVECTOR_ERROR_OUT_OF_MEMORY;
    
    entry->id = id;
    entry->file_offset = file_offset;
    entry->dimension = dimension;
    entry->timestamp = cvector_get_timestamp();
    entry->is_deleted = false;
    entry->next = db->hash_table[hash_idx];
    db->hash_table[hash_idx] = entry;
    
    return CVECTOR_SUCCESS;
}

static cvector_vector_entry_t* cvector_hash_find(cvector_db_t* db, cvector_id_t id) {
    uint64_t hash_idx = cvector_hash(id);
    cvector_vector_entry_t* entry = db->hash_table[hash_idx];
    
    int max_iterations = 1000; // Prevent infinite loops
    while (entry && max_iterations-- > 0) {
        if (entry->id == id && !entry->is_deleted) {
            return entry;
        }
        entry = entry->next;
    }
    
    if (max_iterations <= 0) {
        // Corrupted hash table - log error but don't crash
        fprintf(stderr, "ERROR: Hash table corruption detected for ID %llu\n", 
                (unsigned long long)id);
    }
    
    return NULL;
}

static cvector_error_t cvector_write_header(cvector_db_t* db) {
    cvector_file_header_t header = {0};
    header.magic = CVECTOR_MAGIC_NUMBER;
    header.version = CVECTOR_FILE_VERSION;
    header.dimension = db->config.dimension;
    header.default_similarity = db->config.default_similarity;
    header.vector_count = db->vector_count;
    header.next_id = db->next_id;
    header.created_timestamp = cvector_get_timestamp();
    header.modified_timestamp = header.created_timestamp;
    
    fseek(db->data_file, 0, SEEK_SET);
    size_t written = fwrite(&header, sizeof(header), 1, db->data_file);
    if (written != 1) {
        return CVECTOR_ERROR_FILE_IO;
    }
    
    fflush(db->data_file);
    return CVECTOR_SUCCESS;
}

static cvector_error_t cvector_read_header(cvector_db_t* db) {
    cvector_file_header_t header;
    
    fseek(db->data_file, 0, SEEK_SET);
    size_t read = fread(&header, sizeof(header), 1, db->data_file);
    if (read != 1) {
        return CVECTOR_ERROR_FILE_IO;
    }
    
    if (header.magic != CVECTOR_MAGIC_NUMBER) {
        return CVECTOR_ERROR_DB_CORRUPT;
    }
    
    if (header.version != CVECTOR_FILE_VERSION) {
        return CVECTOR_ERROR_DB_CORRUPT;
    }
    
    db->config.dimension = header.dimension;
    db->config.default_similarity = header.default_similarity;
    db->vector_count = header.vector_count;
    db->next_id = header.next_id;
    
    return CVECTOR_SUCCESS;
}

// Public API Implementation

cvector_error_t cvector_db_create(const cvector_db_config_t* config, cvector_db_t** db) {
    if (!config || !db) {
        return CVECTOR_ERROR_INVALID_ARGS;
    }
    
    if (config->dimension == 0 || config->dimension > CVECTOR_MAX_DIMENSION) {
        return CVECTOR_ERROR_INVALID_ARGS;
    }
    
    // Validate data path
    if (!config->data_path || strlen(config->data_path) == 0) {
        return CVECTOR_ERROR_INVALID_ARGS;
    }
    
    // Validate similarity type
    if (config->default_similarity < CVECTOR_SIMILARITY_COSINE || 
        config->default_similarity > CVECTOR_SIMILARITY_EUCLIDEAN) {
        return CVECTOR_ERROR_INVALID_ARGS;
    }
    
    // Check if database already exists
    struct stat st;
    if (stat(config->data_path, &st) == 0) {
        return CVECTOR_ERROR_FILE_IO;  // Database already exists
    }
    
    // Create database directory if it doesn't exist
    char dir_path[CVECTOR_MAX_PATH];
    strncpy(dir_path, config->data_path, sizeof(dir_path) - 1);
    dir_path[sizeof(dir_path) - 1] = '\0';
    
    char* last_slash = strrchr(dir_path, '/');
    if (last_slash) {
        *last_slash = '\0';
        mkdir(dir_path, 0755);
    }
    
    // Allocate database structure
    *db = calloc(1, sizeof(cvector_db_t));
    if (!*db) {
        return CVECTOR_ERROR_OUT_OF_MEMORY;
    }
    
    cvector_db_t* database = *db;
    memcpy(&database->config, config, sizeof(cvector_db_config_t));
    database->next_id = 1;
    database->vector_count = 0;
    
    // Initialize thread safety mechanisms
    if (pthread_mutex_init(&database->mutex, NULL) != 0) {
        free(database);
        *db = NULL;
        return CVECTOR_ERROR_OUT_OF_MEMORY;
    }
    
    if (pthread_rwlock_init(&database->search_lock, NULL) != 0) {
        pthread_mutex_destroy(&database->mutex);
        free(database);
        *db = NULL;
        return CVECTOR_ERROR_OUT_OF_MEMORY;
    }
    
    // Initialize hash table
    cvector_error_t err = cvector_init_hash_table(database);
    if (err != CVECTOR_SUCCESS) {
        free(database);
        *db = NULL;
        return err;
    }
    
    // Initialize HNSW index
    err = hnsw_create_index(config->dimension, config->default_similarity, &database->hnsw_index);
    if (err != CVECTOR_SUCCESS) {
        cvector_free_hash_table(database);
        free(database);
        *db = NULL;
        return err;
    }
    
    // Create data file
    database->data_file = fopen(config->data_path, "w+b");
    if (!database->data_file) {
        hnsw_destroy_index(database->hnsw_index);
        cvector_free_hash_table(database);
        free(database);
        *db = NULL;
        return CVECTOR_ERROR_FILE_IO;
    }
    
    // Write initial header
    err = cvector_write_header(database);
    if (err != CVECTOR_SUCCESS) {
        fclose(database->data_file);
        hnsw_destroy_index(database->hnsw_index);
        cvector_free_hash_table(database);
        free(database);
        *db = NULL;
        return err;
    }
    
    database->is_open = true;
    return CVECTOR_SUCCESS;
}

cvector_error_t cvector_db_open(const char* db_path, cvector_db_t** db) {
    if (!db_path || !db) {
        return CVECTOR_ERROR_INVALID_ARGS;
    }
    
    // Check if file exists
    struct stat st;
    if (stat(db_path, &st) != 0) {
        return CVECTOR_ERROR_DB_NOT_FOUND;
    }
    
    // Allocate database structure
    *db = calloc(1, sizeof(cvector_db_t));
    if (!*db) {
        return CVECTOR_ERROR_OUT_OF_MEMORY;
    }
    
    cvector_db_t* database = *db;
    strncpy(database->config.data_path, db_path, sizeof(database->config.data_path) - 1);
    database->config.data_path[sizeof(database->config.data_path) - 1] = '\0';
    
    // Initialize hash table
    cvector_error_t err = cvector_init_hash_table(database);
    if (err != CVECTOR_SUCCESS) {
        free(database);
        *db = NULL;
        return err;
    }
    
    // Open data file
    database->data_file = fopen(db_path, "r+b");
    if (!database->data_file) {
        cvector_free_hash_table(database);
        free(database);
        *db = NULL;
        return CVECTOR_ERROR_FILE_IO;
    }
    
    // Read and validate header
    err = cvector_read_header(database);
    if (err != CVECTOR_SUCCESS) {
        fclose(database->data_file);
        cvector_free_hash_table(database);
        free(database);
        *db = NULL;
        return err;
    }
    
    // Initialize HNSW index
    err = hnsw_create_index(database->config.dimension, database->config.default_similarity, &database->hnsw_index);
    if (err != CVECTOR_SUCCESS) {
        fclose(database->data_file);
        cvector_free_hash_table(database);
        free(database);
        *db = NULL;
        return err;
    }
    
    // Rebuild hash table and HNSW index from existing vectors in the file
    fseek(database->data_file, sizeof(cvector_file_header_t), SEEK_SET);
    
    while (true) {
        uint64_t record_start = ftell(database->data_file);
        cvector_vector_record_t record;
        size_t read = fread(&record, sizeof(record), 1, database->data_file);
        if (read != 1) break; // End of file or error
        
        if (!record.is_deleted) {
            // Read vector data
            float* vector_data = malloc(record.dimension * sizeof(float));
            if (vector_data) {
                read = fread(vector_data, sizeof(float), record.dimension, database->data_file);
                if (read == record.dimension) {
                    // Add to hash table
                    cvector_hash_insert(database, record.id, record_start, record.dimension);
                    
                    // Rebuild HNSW index - add vector back to HNSW
                    if (database->hnsw_index) {
                        cvector_error_t hnsw_err = hnsw_add_vector(database->hnsw_index, record.id, vector_data);
                        if (hnsw_err != CVECTOR_SUCCESS) {
                            printf("Warning: Failed to rebuild HNSW vector %llu: %s\n", 
                                   (unsigned long long)record.id, cvector_error_string(hnsw_err));
                        }
                    }
                }
                free(vector_data);
            }
        } else {
            // Skip deleted vector data
            fseek(database->data_file, record.dimension * sizeof(float), SEEK_CUR);
        }
    }
    
    database->is_open = true;
    return CVECTOR_SUCCESS;
}

cvector_error_t cvector_db_close(cvector_db_t* db) {
    if (!db || !db->is_open) {
        return CVECTOR_ERROR_INVALID_ARGS;
    }
    
    // Update header with final stats
    cvector_write_header(db);
    
    // Close files
    if (db->data_file) {
        fclose(db->data_file);
        db->data_file = NULL;
    }
    
    // Free hash table
    cvector_free_hash_table(db);
    
    // Destroy HNSW index
    if (db->hnsw_index) {
        hnsw_destroy_index(db->hnsw_index);
        db->hnsw_index = NULL;
    }
    
    // Cleanup thread safety mechanisms
    pthread_mutex_destroy(&db->mutex);
    pthread_rwlock_destroy(&db->search_lock);
    
    db->is_open = false;
    free(db);
    
    return CVECTOR_SUCCESS;
}

cvector_error_t cvector_db_drop(const char* db_path) {
    if (!db_path) {
        return CVECTOR_ERROR_INVALID_ARGS;
    }
    
    if (unlink(db_path) != 0) {
        return CVECTOR_ERROR_FILE_IO;
    }
    
    return CVECTOR_SUCCESS;
}

cvector_error_t cvector_insert(cvector_db_t* db, const cvector_t* vector) {
    if (!db || !db->is_open || !vector || !vector->data) {
        return CVECTOR_ERROR_INVALID_ARGS;
    }
    
    if (vector->dimension != db->config.dimension) {
        return CVECTOR_ERROR_DIMENSION_MISMATCH;
    }
    
    // Thread safety: acquire write lock
    pthread_mutex_lock(&db->mutex);
    
    // Check if vector with this ID already exists
    if (cvector_hash_find(db, vector->id)) {
        pthread_mutex_unlock(&db->mutex);
        return CVECTOR_ERROR_INVALID_ARGS;  // Vector already exists
    }
    
    // Seek to end of file
    fseek(db->data_file, 0, SEEK_END);
    uint64_t file_offset = ftell(db->data_file);
    
    // Create record
    cvector_vector_record_t record = {0};
    record.id = vector->id;
    record.dimension = vector->dimension;
    record.timestamp = cvector_get_timestamp();
    record.is_deleted = 0;
    
    // Write record header
    size_t written = fwrite(&record, sizeof(record), 1, db->data_file);
    if (written != 1) {
        pthread_mutex_unlock(&db->mutex);
        return CVECTOR_ERROR_FILE_IO;
    }
    
    // Write vector data
    written = fwrite(vector->data, sizeof(float), vector->dimension, db->data_file);
    if (written != vector->dimension) {
        pthread_mutex_unlock(&db->mutex);
        return CVECTOR_ERROR_FILE_IO;
    }
    
    // Add to hash table
    cvector_error_t err = cvector_hash_insert(db, vector->id, file_offset, vector->dimension);
    if (err != CVECTOR_SUCCESS) {
        pthread_mutex_unlock(&db->mutex);
        return err;
    }
    
    // Add to HNSW index
    if (db->hnsw_index) {
        err = hnsw_add_vector(db->hnsw_index, vector->id, vector->data);
        if (err != CVECTOR_SUCCESS) {
            // Note: In production, we might want to rollback the hash table entry
            // For now, we'll log the error but continue
            printf("Warning: Failed to add vector %llu to HNSW index: %s\n", 
                   vector->id, cvector_error_string(err));
        }
    }
    
    // Update counters
    db->vector_count++;
    if (vector->id >= db->next_id) {
        db->next_id = vector->id + 1;
    }
    
    fflush(db->data_file);
    
    // Thread safety: release write lock
    pthread_mutex_unlock(&db->mutex);
    
    return CVECTOR_SUCCESS;
}

cvector_error_t cvector_get(cvector_db_t* db, cvector_id_t id, cvector_t** vector) {
    // Comprehensive input validation
    if (!db) {
        return CVECTOR_ERROR_INVALID_ARGS;
    }
    
    if (!db->is_open) {
        return CVECTOR_ERROR_DB_NOT_FOUND;
    }
    
    if (!vector) {
        return CVECTOR_ERROR_INVALID_ARGS;
    }
    
    if (id == 0) {  // ID 0 is typically invalid
        return CVECTOR_ERROR_INVALID_ARGS;
    }
    
    // Find in hash table
    cvector_vector_entry_t* entry = cvector_hash_find(db, id);
    if (!entry) {
        return CVECTOR_ERROR_VECTOR_NOT_FOUND;
    }
    
    // Seek to record position
    fseek(db->data_file, entry->file_offset, SEEK_SET);
    
    // Read record header
    cvector_vector_record_t record;
    size_t read = fread(&record, sizeof(record), 1, db->data_file);
    if (read != 1) {
        return CVECTOR_ERROR_FILE_IO;
    }
    
    if (record.is_deleted) {
        return CVECTOR_ERROR_VECTOR_NOT_FOUND;
    }
    
    // Allocate vector
    cvector_t* result = malloc(sizeof(cvector_t));
    if (!result) {
        return CVECTOR_ERROR_OUT_OF_MEMORY;
    }
    
    result->data = malloc(record.dimension * sizeof(float));
    if (!result->data) {
        free(result);
        return CVECTOR_ERROR_OUT_OF_MEMORY;
    }
    
    // Read vector data
    read = fread(result->data, sizeof(float), record.dimension, db->data_file);
    if (read != record.dimension) {
        free(result->data);
        free(result);
        return CVECTOR_ERROR_FILE_IO;
    }
    
    result->id = record.id;
    result->dimension = record.dimension;
    result->timestamp = record.timestamp;
    
    *vector = result;
    return CVECTOR_SUCCESS;
}

cvector_error_t cvector_delete(cvector_db_t* db, cvector_id_t id) {
    // Comprehensive input validation
    if (!db) {
        return CVECTOR_ERROR_INVALID_ARGS;
    }
    
    if (!db->is_open) {
        return CVECTOR_ERROR_DB_NOT_FOUND;
    }
    
    if (id == 0) {  // ID 0 is typically invalid
        return CVECTOR_ERROR_INVALID_ARGS;
    }
    
    // Thread safety: acquire write lock
    pthread_mutex_lock(&db->mutex);
    
    // Find in hash table
    cvector_vector_entry_t* entry = cvector_hash_find(db, id);
    if (!entry) {
        pthread_mutex_unlock(&db->mutex);
        return CVECTOR_ERROR_VECTOR_NOT_FOUND;
    }
    
    // Mark as deleted in hash table
    entry->is_deleted = true;
    
    // Remove from HNSW index
    if (db->hnsw_index) {
        cvector_error_t hnsw_err = hnsw_remove_vector(db->hnsw_index, id);
        if (hnsw_err != CVECTOR_SUCCESS) {
            printf("Warning: Failed to remove vector %llu from HNSW index: %s\n", 
                   id, cvector_error_string(hnsw_err));
        }
    }
    
    // Mark as deleted in file
    fseek(db->data_file, entry->file_offset + offsetof(cvector_vector_record_t, is_deleted), SEEK_SET);
    uint8_t deleted_flag = 1;
    size_t written = fwrite(&deleted_flag, sizeof(deleted_flag), 1, db->data_file);
    if (written != 1) {
        pthread_mutex_unlock(&db->mutex);
        return CVECTOR_ERROR_FILE_IO;
    }
    
    db->vector_count--;
    fflush(db->data_file);
    
    // Thread safety: release write lock
    pthread_mutex_unlock(&db->mutex);
    
    return CVECTOR_SUCCESS;
}

cvector_error_t cvector_search(cvector_db_t* db, const cvector_query_t* query, 
                              cvector_result_t** results, size_t* result_count) {
    if (!db || !db->is_open || !query || !results || !result_count) {
        return CVECTOR_ERROR_INVALID_ARGS;
    }
    
    if (!query->query_vector || query->dimension != db->config.dimension) {
        return CVECTOR_ERROR_INVALID_ARGS;
    }
    
    // Validate query parameters
    if (query->top_k == 0 || query->top_k > 10000) {  // Reasonable limit
        return CVECTOR_ERROR_INVALID_ARGS;
    }
    
    if (query->min_similarity < -1.0f || query->min_similarity > 1.0f) {
        return CVECTOR_ERROR_INVALID_ARGS;
    }
    
    // Thread safety: acquire read lock for search operations
    pthread_rwlock_rdlock(&db->search_lock);
    
    *results = NULL;
    *result_count = 0;
    
    // If empty index, return empty results
    if (db->vector_count == 0) {
        return CVECTOR_SUCCESS;
    }
    
    // Try HNSW search first, fall back to brute force if needed
    if (db->hnsw_index && db->vector_count > 0) {
        hnsw_search_result_t* hnsw_result = NULL;
        cvector_error_t hnsw_err = hnsw_search_with_ef(db->hnsw_index, query->query_vector, 
                                                       query->top_k, query->top_k * 2, &hnsw_result);
        
        if (hnsw_err == CVECTOR_SUCCESS && hnsw_result && hnsw_result->count > 0) {
            // Convert HNSW results to our format
            *results = malloc(hnsw_result->count * sizeof(cvector_result_t));
            if (*results) {
                *result_count = 0;
                for (uint32_t i = 0; i < hnsw_result->count && *result_count < query->top_k; i++) {
                    float similarity = hnsw_result->similarities[i];
                    
                    // Apply similarity threshold
                    if (query->min_similarity == 0.0f || similarity >= query->min_similarity) {
                        (*results)[*result_count].id = hnsw_result->ids[i];
                        (*results)[*result_count].similarity = similarity;
                        (*results)[*result_count].vector = NULL;
                        (*result_count)++;
                    }
                }
                
                // Resize results array if needed
                if (*result_count < hnsw_result->count) {
                    cvector_result_t* resized = realloc(*results, *result_count * sizeof(cvector_result_t));
                    if (resized) {
                        *results = resized;
                    }
                }
                
                hnsw_free_search_result(hnsw_result);
                return CVECTOR_SUCCESS;
            }
        }
        
        if (hnsw_result) {
            hnsw_free_search_result(hnsw_result);
        }
        
        // If HNSW failed, fall back to brute force
        printf("HNSW search failed, falling back to brute force\n");
    }
    
    // Brute force search fallback
    size_t max_results = (query->top_k < db->vector_count) ? query->top_k : db->vector_count;
    cvector_result_t* temp_results = malloc(max_results * sizeof(cvector_result_t));
    if (!temp_results) {
        pthread_rwlock_unlock(&db->search_lock);
        return CVECTOR_ERROR_OUT_OF_MEMORY;
    }
    
    size_t valid_results = 0;
    
    // Iterate through all vectors and calculate similarities
    for (size_t i = 0; i < db->hash_table_size && valid_results < max_results; i++) {
        cvector_vector_entry_t* entry = db->hash_table[i];
        
        while (entry && valid_results < max_results) {
            if (!entry->is_deleted) {
                // Get the vector data
                cvector_t* vector = NULL;
                cvector_error_t get_err = cvector_get(db, entry->id, &vector);
                if (get_err == CVECTOR_SUCCESS && vector) {
                    // Calculate similarity
                    float similarity = 0.0f;
                    switch (query->similarity) {
                        case CVECTOR_SIMILARITY_COSINE:
                            similarity = cvector_cosine_similarity(query->query_vector, vector->data, query->dimension);
                            break;
                        case CVECTOR_SIMILARITY_DOT_PRODUCT:
                            similarity = cvector_dot_product(query->query_vector, vector->data, query->dimension);
                            break;
                        case CVECTOR_SIMILARITY_EUCLIDEAN:
                            similarity = -cvector_euclidean_distance(query->query_vector, vector->data, query->dimension);
                            break;
                        default:
                            similarity = 0.0f;
                    }
                    
                    // Check minimum similarity threshold
                    if (query->min_similarity == 0.0f || similarity >= query->min_similarity) {
                        temp_results[valid_results].id = entry->id;
                        temp_results[valid_results].similarity = similarity;
                        temp_results[valid_results].vector = NULL;
                        valid_results++;
                    }
                    
                    cvector_free_vector(vector);
                }
            }
            entry = entry->next;
        }
    }
    
    // Sort results by similarity (descending) - only if we have results
    if (valid_results > 1) {
        for (size_t i = 0; i < valid_results - 1; i++) {
            for (size_t j = 0; j < valid_results - i - 1; j++) {
                if (temp_results[j].similarity < temp_results[j + 1].similarity) {
                    cvector_result_t temp = temp_results[j];
                    temp_results[j] = temp_results[j + 1];
                    temp_results[j + 1] = temp;
                }
            }
        }
    }
    
    // Limit to top_k results
    if (valid_results > query->top_k) {
        valid_results = query->top_k;
    }
    
    if (valid_results == 0) {
        free(temp_results);
        pthread_rwlock_unlock(&db->search_lock);
        return CVECTOR_SUCCESS;
    }
    
    // Resize results array to actual size
    if (valid_results < max_results) {
        cvector_result_t* final_results = realloc(temp_results, valid_results * sizeof(cvector_result_t));
        if (final_results) {
            temp_results = final_results;
        }
    }
    
    *results = temp_results;
    *result_count = valid_results;
    
    // Thread safety: release read lock
    pthread_rwlock_unlock(&db->search_lock);
    
    return CVECTOR_SUCCESS;
}

// Utility functions

cvector_error_t cvector_create_vector(cvector_id_t id, uint32_t dimension, 
                                     const float* data, cvector_t** vector) {
    if (!data || !vector || dimension == 0) {
        return CVECTOR_ERROR_INVALID_ARGS;
    }
    
    cvector_t* v = malloc(sizeof(cvector_t));
    if (!v) {
        return CVECTOR_ERROR_OUT_OF_MEMORY;
    }
    
    v->data = malloc(dimension * sizeof(float));
    if (!v->data) {
        free(v);
        return CVECTOR_ERROR_OUT_OF_MEMORY;
    }
    
    v->id = id;
    v->dimension = dimension;
    v->timestamp = cvector_get_timestamp();
    memcpy(v->data, data, dimension * sizeof(float));
    
    *vector = v;
    return CVECTOR_SUCCESS;
}

void cvector_free_vector(cvector_t* vector) {
    if (vector) {
        free(vector->data);
        free(vector);
    }
}

void cvector_free_results(cvector_result_t* results, size_t count) {
    if (results) {
        for (size_t i = 0; i < count; i++) {
            if (results[i].vector) {
                cvector_free_vector(results[i].vector);
            }
        }
        free(results);
    }
}

const char* cvector_error_string(cvector_error_t error) {
    switch (error) {
        case CVECTOR_SUCCESS: return "Success";
        case CVECTOR_ERROR_INVALID_ARGS: return "Invalid arguments";
        case CVECTOR_ERROR_OUT_OF_MEMORY: return "Out of memory";
        case CVECTOR_ERROR_FILE_IO: return "File I/O error";
        case CVECTOR_ERROR_DB_NOT_FOUND: return "Database not found";
        case CVECTOR_ERROR_VECTOR_NOT_FOUND: return "Vector not found";
        case CVECTOR_ERROR_DIMENSION_MISMATCH: return "Dimension mismatch";
        case CVECTOR_ERROR_DB_CORRUPT: return "Database corrupt";
        default: return "Unknown error";
    }
}

cvector_error_t cvector_db_stats(cvector_db_t* db, cvector_db_stats_t* stats) {
    if (!db || !db->is_open || !stats) {
        return CVECTOR_ERROR_INVALID_ARGS;
    }
    
    stats->total_vectors = db->vector_count;
    stats->dimension = db->config.dimension;
    stats->default_similarity = db->config.default_similarity;
    strncpy(stats->db_path, db->config.data_path, sizeof(stats->db_path) - 1);
    stats->db_path[sizeof(stats->db_path) - 1] = '\0';
    
    // Calculate total size
    fseek(db->data_file, 0, SEEK_END);
    stats->total_size_bytes = ftell(db->data_file);
    
    return CVECTOR_SUCCESS;
}