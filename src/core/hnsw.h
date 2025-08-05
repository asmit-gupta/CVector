#ifndef CVECTOR_HNSW_H
#define CVECTOR_HNSW_H

#include "cvector.h"
#include <stdbool.h>
#include <stdint.h>
#include <pthread.h>

// HNSW Configuration
#define HNSW_DEFAULT_M 16                  // Max connections per node
#define HNSW_DEFAULT_EF_CONSTRUCTION 200   // Construction time accuracy
#define HNSW_DEFAULT_EF_SEARCH 50          // Search time accuracy
#define HNSW_DEFAULT_ML 1.0 / log(2.0)     // Level generation factor
#define HNSW_MAX_LEVEL 16                  // Maximum number of levels

// HNSW Node structure
typedef struct hnsw_node {
    cvector_id_t id;                       // Vector ID
    uint32_t level;                        // Node level (0 = base layer)
    uint32_t* connections[HNSW_MAX_LEVEL]; // Connections per level
    uint32_t connection_count[HNSW_MAX_LEVEL]; // Number of connections per level
    float* vector_data;                    // Cached vector for fast similarity
    uint32_t dimension;                    // Vector dimension
} hnsw_node_t;

// HNSW Index structure
typedef struct hnsw_index {
    hnsw_node_t** nodes;                   // Array of nodes
    uint32_t node_count;                   // Current number of nodes
    uint32_t node_capacity;                // Allocated capacity
    uint32_t entry_point;                  // Top-level entry point
    uint32_t max_level;                    // Current maximum level
    
    // Configuration
    uint32_t M;                            // Max connections per node
    uint32_t ef_construction;              // Construction accuracy
    uint32_t ef_search;                    // Search accuracy
    float ml;                              // Level generation factor
    uint32_t dimension;                    // Vector dimension
    cvector_similarity_t similarity_type;  // Similarity metric
    
    // Thread Safety
    pthread_mutex_t write_mutex;           // Protects structure modifications
    pthread_rwlock_t search_lock;          // Read-write lock for search operations
    volatile bool is_corrupted;            // Corruption detection flag
    
    // Statistics (atomic for thread safety)
    volatile uint64_t search_count;        // Number of searches performed
    volatile uint64_t total_distance_computations;  // Performance tracking
    volatile uint64_t insert_count;        // Number of insertions
    volatile uint64_t delete_count;        // Number of deletions
    
    // Memory Management
    void* memory_pool;                     // Optional memory pool for allocations
    size_t memory_pool_size;               // Size of memory pool
    volatile uint64_t memory_used;         // Current memory usage
    
    // Integrity checking
    uint32_t checksum;                     // Simple integrity checksum
    uint64_t last_modified;                // Last modification timestamp
} hnsw_index_t;

// Search result structure
typedef struct hnsw_search_result {
    cvector_id_t* ids;                     // Result vector IDs
    float* similarities;                   // Similarity scores
    uint32_t count;                        // Number of results
    uint32_t capacity;                     // Allocated capacity
} hnsw_search_result_t;

// Priority queue item
typedef struct {
    uint32_t node_id;
    float distance;
} hnsw_pq_item_t;

// Priority queue for search
typedef struct hnsw_priority_queue {
    hnsw_pq_item_t* items;
    uint32_t count;
    uint32_t capacity;
    bool is_max_heap;                      // true for max-heap, false for min-heap
} hnsw_priority_queue_t;

// HNSW Index Operations
cvector_error_t hnsw_create_index(uint32_t dimension, cvector_similarity_t similarity_type, 
                                  hnsw_index_t** index);
cvector_error_t hnsw_destroy_index(hnsw_index_t* index);

// Node Operations
cvector_error_t hnsw_add_vector(hnsw_index_t* index, cvector_id_t id, const float* vector);
cvector_error_t hnsw_remove_vector(hnsw_index_t* index, cvector_id_t id);

// Search Operations
cvector_error_t hnsw_search(hnsw_index_t* index, const float* query_vector, 
                           uint32_t top_k, hnsw_search_result_t** result);
cvector_error_t hnsw_search_with_ef(hnsw_index_t* index, const float* query_vector,
                                   uint32_t top_k, uint32_t ef, hnsw_search_result_t** result);

// Utility Functions
void hnsw_free_search_result(hnsw_search_result_t* result);

// Priority Queue Operations (internal)
cvector_error_t hnsw_pq_create(uint32_t capacity, bool is_max_heap, hnsw_priority_queue_t** pq);
void hnsw_pq_destroy(hnsw_priority_queue_t* pq);
cvector_error_t hnsw_pq_push(hnsw_priority_queue_t* pq, uint32_t node_id, float distance);
bool hnsw_pq_pop(hnsw_priority_queue_t* pq, uint32_t* node_id, float* distance);
bool hnsw_pq_is_empty(hnsw_priority_queue_t* pq);
bool hnsw_pq_is_full(hnsw_priority_queue_t* pq);

// Similarity Functions
float hnsw_calculate_similarity(const float* a, const float* b, uint32_t dimension, 
                               cvector_similarity_t similarity_type);

// Configuration
typedef struct hnsw_config {
    uint32_t M;                           // Max connections (default: 16)
    uint32_t ef_construction;             // Construction accuracy (default: 200)
    uint32_t ef_search;                   // Search accuracy (default: 50)
    float ml;                             // Level generation factor
} hnsw_config_t;

cvector_error_t hnsw_set_config(hnsw_index_t* index, const hnsw_config_t* config);
cvector_error_t hnsw_get_config(hnsw_index_t* index, hnsw_config_t* config);

// Statistics
typedef struct {
    uint32_t node_count;                  // Total nodes in index
    uint32_t max_level;                   // Maximum level
    uint64_t search_count;                // Number of searches
    uint64_t distance_computations;       // Total distance calculations
    float avg_connections_per_node;       // Average connections
    uint32_t entry_point_level;           // Entry point level
} hnsw_stats_t;

// Statistics Functions
cvector_error_t hnsw_get_stats(hnsw_index_t* index, hnsw_stats_t* stats);

// Persistence
cvector_error_t hnsw_save_index(hnsw_index_t* index, const char* filepath);
cvector_error_t hnsw_load_index(const char* filepath, hnsw_index_t** index);

// Production-grade Integrity and Recovery
cvector_error_t hnsw_validate_integrity(hnsw_index_t* index);
cvector_error_t hnsw_repair_index(hnsw_index_t* index);
cvector_error_t hnsw_backup_index(hnsw_index_t* index, const char* backup_path);
cvector_error_t hnsw_restore_from_backup(const char* backup_path, hnsw_index_t** index);

// Advanced Statistics and Monitoring
typedef struct {
    uint32_t node_count;
    uint32_t max_level;
    uint64_t search_count;
    uint64_t insert_count;
    uint64_t delete_count;
    uint64_t distance_computations;
    float avg_connections_per_node;
    uint32_t entry_point_level;
    uint64_t memory_used;
    uint64_t memory_pool_size;
    double avg_search_time_ms;
    double avg_insert_time_ms;
    bool is_corrupted;
    uint64_t last_modified;
} hnsw_detailed_stats_t;

cvector_error_t hnsw_get_detailed_stats(hnsw_index_t* index, hnsw_detailed_stats_t* stats);

// Thread Safety Control
cvector_error_t hnsw_lock_for_write(hnsw_index_t* index);
cvector_error_t hnsw_unlock_write(hnsw_index_t* index);
cvector_error_t hnsw_lock_for_read(hnsw_index_t* index);
cvector_error_t hnsw_unlock_read(hnsw_index_t* index);

// Memory Management
cvector_error_t hnsw_init_memory_pool(hnsw_index_t* index, size_t pool_size);
cvector_error_t hnsw_cleanup_memory_pool(hnsw_index_t* index);

// Performance Monitoring
typedef struct {
    uint64_t start_time_ns;
    uint64_t end_time_ns;
    uint32_t operation_type;  // 0=search, 1=insert, 2=delete
    cvector_error_t result;
} hnsw_perf_record_t;

cvector_error_t hnsw_start_perf_monitoring(hnsw_index_t* index);
cvector_error_t hnsw_stop_perf_monitoring(hnsw_index_t* index);

#endif // CVECTOR_HNSW_H