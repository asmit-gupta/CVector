#ifndef CVECTOR_HNSW_H
#define CVECTOR_HNSW_H

#include "cvector.h"
#include <stdbool.h>
#include <stdint.h>

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
    
    // Statistics
    uint64_t search_count;                 // Number of searches performed
    uint64_t total_distance_computations;  // Performance tracking
} hnsw_index_t;

// Search result structure
typedef struct hnsw_search_result {
    cvector_id_t* ids;                     // Result vector IDs
    float* similarities;                   // Similarity scores
    uint32_t count;                        // Number of results
    uint32_t capacity;                     // Allocated capacity
} hnsw_search_result_t;

// Priority queue for search
typedef struct hnsw_priority_queue {
    struct {
        uint32_t node_id;
        float distance;
    } *items;
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
cvector_error_t hnsw_get_stats(hnsw_index_t* index, struct hnsw_stats* stats);
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
typedef struct hnsw_stats {
    uint32_t node_count;                  // Total nodes in index
    uint32_t max_level;                   // Maximum level
    uint64_t search_count;                // Number of searches
    uint64_t distance_computations;       // Total distance calculations
    float avg_connections_per_node;       // Average connections
    uint32_t entry_point_level;           // Entry point level
} hnsw_stats_t;

// Persistence (future)
cvector_error_t hnsw_save_index(hnsw_index_t* index, const char* filepath);
cvector_error_t hnsw_load_index(const char* filepath, hnsw_index_t** index);

#endif // CVECTOR_HNSW_H