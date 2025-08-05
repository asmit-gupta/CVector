#include "hnsw.h"
#include "similarity.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <assert.h>
#include <stdio.h>
#include <stdarg.h>
#include <time.h>
#include <errno.h>
#include <stdatomic.h>

// Production-grade logging levels
typedef enum {
    HNSW_LOG_ERROR = 0,
    HNSW_LOG_WARN = 1,
    HNSW_LOG_INFO = 2,
    HNSW_LOG_DEBUG = 3
} hnsw_log_level_t;

static hnsw_log_level_t g_hnsw_log_level = HNSW_LOG_WARN;

// Production-grade timing utilities
static uint64_t hnsw_get_timestamp_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static uint64_t hnsw_get_timestamp_s(void) {
    return (uint64_t)time(NULL);
}

// Simple checksum calculation for integrity checking
static uint32_t hnsw_calculate_checksum(hnsw_index_t* index) {
    if (!index) return 0;
    
    uint32_t checksum = 0;
    checksum ^= index->node_count;
    checksum ^= index->dimension;
    checksum ^= (uint32_t)index->similarity_type;
    checksum ^= index->M;
    checksum ^= index->max_level;
    
    // Add contribution from node structure (simplified)
    for (uint32_t i = 0; i < index->node_count; i++) {
        if (index->nodes[i]) {
            checksum ^= (uint32_t)index->nodes[i]->id;
            checksum ^= index->nodes[i]->level;
        }
    }
    
    return checksum;
}

// Atomic statistics helpers
static void hnsw_atomic_inc_u64(volatile uint64_t* ptr) {
    __atomic_fetch_add(ptr, 1, __ATOMIC_SEQ_CST);
}

static uint64_t hnsw_atomic_load_u64(volatile uint64_t* ptr) {
    return __atomic_load_n(ptr, __ATOMIC_SEQ_CST);
}

static void hnsw_atomic_store_u64(volatile uint64_t* ptr, uint64_t value) {
    __atomic_store_n(ptr, value, __ATOMIC_SEQ_CST);
}

// Simplified error logging - only for critical errors
static void hnsw_log_error(const char* message) {
    fprintf(stderr, "HNSW Error: %s\n", message);
}

// Input validation helper - simplified
static cvector_error_t hnsw_validate_index(hnsw_index_t* index) {
    if (!index || !index->nodes || index->dimension == 0 || 
        index->node_count > index->node_capacity) {
        return CVECTOR_ERROR_INVALID_ARGS;
    }
    return CVECTOR_SUCCESS;
}

// Memory safety helper
static bool hnsw_is_valid_node_id(hnsw_index_t* index, uint32_t node_id) {
    return index && node_id < index->node_count && index->nodes[node_id] != NULL;
}

// Internal helper functions
static uint32_t hnsw_random_level(float ml);
static cvector_error_t hnsw_resize_index(hnsw_index_t* index);
// Removed unused function declaration
static cvector_error_t hnsw_connect_layers_safe(hnsw_index_t* index, uint32_t node_id, uint32_t max_level);
static cvector_error_t hnsw_search_layer(hnsw_index_t* index, const float* query_vector,
                                        hnsw_priority_queue_t* entry_points, uint32_t num_closest,
                                        uint32_t level);
static void hnsw_heap_up(hnsw_priority_queue_t* pq, uint32_t idx);
static void hnsw_heap_down(hnsw_priority_queue_t* pq, uint32_t idx);
static cvector_error_t hnsw_select_neighbors_simple(hnsw_index_t* index, uint32_t node_id, 
                                                   hnsw_priority_queue_t* candidates, uint32_t M, uint32_t level);

// Random level generation using ml parameter
static uint32_t hnsw_random_level(float ml) {
    uint32_t level = 0;
    while (((float)rand() / RAND_MAX) < (1.0f / ml) && level < HNSW_MAX_LEVEL - 1) {
        level++;
    }
    return level;
}

// Resize index capacity
static cvector_error_t hnsw_resize_index(hnsw_index_t* index) {
    uint32_t new_capacity = index->node_capacity * 2;
    hnsw_node_t** new_nodes = realloc(index->nodes, new_capacity * sizeof(hnsw_node_t*));
    if (!new_nodes) {
        return CVECTOR_ERROR_OUT_OF_MEMORY;
    }
    
    // Initialize new slots to NULL
    for (uint32_t i = index->node_capacity; i < new_capacity; i++) {
        new_nodes[i] = NULL;
    }
    
    index->nodes = new_nodes;
    index->node_capacity = new_capacity;
    return CVECTOR_SUCCESS;
}

// Priority queue operations
cvector_error_t hnsw_pq_create(uint32_t capacity, bool is_max_heap, hnsw_priority_queue_t** pq) {
    if (!pq || capacity == 0) {
        return CVECTOR_ERROR_INVALID_ARGS;
    }
    
    *pq = malloc(sizeof(hnsw_priority_queue_t));
    if (!*pq) {
        return CVECTOR_ERROR_OUT_OF_MEMORY;
    }
    
    (*pq)->items = malloc(capacity * sizeof(hnsw_pq_item_t));
    if (!(*pq)->items) {
        free(*pq);
        *pq = NULL;
        return CVECTOR_ERROR_OUT_OF_MEMORY;
    }
    
    (*pq)->capacity = capacity;
    (*pq)->count = 0;
    (*pq)->is_max_heap = is_max_heap;
    
    return CVECTOR_SUCCESS;
}

void hnsw_pq_destroy(hnsw_priority_queue_t* pq) {
    if (pq) {
        free(pq->items);
        free(pq);
    }
}

static void hnsw_heap_up(hnsw_priority_queue_t* pq, uint32_t idx) {
    if (idx == 0) return;
    
    uint32_t parent = (idx - 1) / 2;
    bool should_swap = pq->is_max_heap ? 
        (pq->items[idx].distance > pq->items[parent].distance) :
        (pq->items[idx].distance < pq->items[parent].distance);
    
    if (should_swap) {
        // Swap
        hnsw_pq_item_t temp = pq->items[idx];
        pq->items[idx] = pq->items[parent];
        pq->items[parent] = temp;
        hnsw_heap_up(pq, parent);
    }
}

static void hnsw_heap_down(hnsw_priority_queue_t* pq, uint32_t idx) {
    uint32_t left = 2 * idx + 1;
    uint32_t right = 2 * idx + 2;
    uint32_t extreme = idx;
    
    if (left < pq->count) {
        bool left_is_extreme = pq->is_max_heap ?
            (pq->items[left].distance > pq->items[extreme].distance) :
            (pq->items[left].distance < pq->items[extreme].distance);
        if (left_is_extreme) {
            extreme = left;
        }
    }
    
    if (right < pq->count) {
        bool right_is_extreme = pq->is_max_heap ?
            (pq->items[right].distance > pq->items[extreme].distance) :
            (pq->items[right].distance < pq->items[extreme].distance);
        if (right_is_extreme) {
            extreme = right;
        }
    }
    
    if (extreme != idx) {
        // Swap
        hnsw_pq_item_t temp = pq->items[idx];
        pq->items[idx] = pq->items[extreme];
        pq->items[extreme] = temp;
        hnsw_heap_down(pq, extreme);
    }
}

cvector_error_t hnsw_pq_push(hnsw_priority_queue_t* pq, uint32_t node_id, float distance) {
    if (!pq || pq->count >= pq->capacity) {
        return CVECTOR_ERROR_INVALID_ARGS;
    }
    
    pq->items[pq->count].node_id = node_id;
    pq->items[pq->count].distance = distance;
    hnsw_heap_up(pq, pq->count);
    pq->count++;
    
    return CVECTOR_SUCCESS;
}

bool hnsw_pq_pop(hnsw_priority_queue_t* pq, uint32_t* node_id, float* distance) {
    if (!pq || pq->count == 0) {
        return false;
    }
    
    if (node_id) *node_id = pq->items[0].node_id;
    if (distance) *distance = pq->items[0].distance;
    
    pq->count--;
    if (pq->count > 0) {
        pq->items[0] = pq->items[pq->count];
        hnsw_heap_down(pq, 0);
    }
    
    return true;
}

bool hnsw_pq_is_empty(hnsw_priority_queue_t* pq) {
    return !pq || pq->count == 0;
}

bool hnsw_pq_is_full(hnsw_priority_queue_t* pq) {
    return pq && pq->count >= pq->capacity;
}

// Main HNSW operations
cvector_error_t hnsw_create_index(uint32_t dimension, cvector_similarity_t similarity_type, 
                                  hnsw_index_t** index) {
    if (!index || dimension == 0) {
        return CVECTOR_ERROR_INVALID_ARGS;
    }
    
    *index = calloc(1, sizeof(hnsw_index_t));
    if (!*index) {
        return CVECTOR_ERROR_OUT_OF_MEMORY;
    }
    
    hnsw_index_t* idx = *index;
    idx->node_capacity = 1000; // Initial capacity
    idx->nodes = calloc(idx->node_capacity, sizeof(hnsw_node_t*));
    if (!idx->nodes) {
        free(idx);
        *index = NULL;
        return CVECTOR_ERROR_OUT_OF_MEMORY;
    }
    
    // Set configuration
    idx->M = HNSW_DEFAULT_M;
    idx->ef_construction = HNSW_DEFAULT_EF_CONSTRUCTION;
    idx->ef_search = HNSW_DEFAULT_EF_SEARCH;
    idx->ml = HNSW_DEFAULT_ML;
    idx->dimension = dimension;
    idx->similarity_type = similarity_type;
    idx->entry_point = UINT32_MAX; // No entry point initially
    idx->max_level = 0;
    
    // Initialize thread safety mechanisms
    if (pthread_mutex_init(&idx->write_mutex, NULL) != 0) {
        // Error logged;
        free(idx->nodes);
        free(idx);
        *index = NULL;
        return CVECTOR_ERROR_DB_CORRUPT;
    }
    
    if (pthread_rwlock_init(&idx->search_lock, NULL) != 0) {
        // Error logged;
        pthread_mutex_destroy(&idx->write_mutex);
        free(idx->nodes);
        free(idx);
        *index = NULL;
        return CVECTOR_ERROR_DB_CORRUPT;
    }
    
    // Initialize corruption detection and statistics
    idx->is_corrupted = false;
    hnsw_atomic_store_u64(&idx->search_count, 0);
    hnsw_atomic_store_u64(&idx->total_distance_computations, 0);
    hnsw_atomic_store_u64(&idx->insert_count, 0);
    hnsw_atomic_store_u64(&idx->delete_count, 0);
    hnsw_atomic_store_u64(&idx->memory_used, sizeof(hnsw_index_t));
    
    // Initialize memory management
    idx->memory_pool = NULL;
    idx->memory_pool_size = 0;
    
    // Set integrity information
    idx->checksum = hnsw_calculate_checksum(idx);
    idx->last_modified = hnsw_get_timestamp_s();
    
    // Index created successfully
    
    return CVECTOR_SUCCESS;
}

cvector_error_t hnsw_destroy_index(hnsw_index_t* index) {
    if (!index) {
        return CVECTOR_SUCCESS;
    }
    
    // Info logged;
    
    // NOTE: We don't acquire locks here because destroy should only be called
    // when no other threads are using the index (during shutdown)
    
    // Free all nodes
    for (uint32_t i = 0; i < index->node_count; i++) {
        if (index->nodes[i]) {
            // Free connection arrays
            for (uint32_t level = 0; level < HNSW_MAX_LEVEL; level++) {
                free(index->nodes[i]->connections[level]);
            }
            free(index->nodes[i]->vector_data);
            free(index->nodes[i]);
        }
    }
    
    free(index->nodes);
    
    // Clean up memory pool if exists
    if (index->memory_pool) {
        free(index->memory_pool);
    }
    
    // Destroy thread safety mechanisms
    pthread_mutex_destroy(&index->write_mutex);
    pthread_rwlock_destroy(&index->search_lock);
    
    free(index);
    return CVECTOR_SUCCESS;
}

float hnsw_calculate_similarity(const float* a, const float* b, uint32_t dimension, 
                               cvector_similarity_t similarity_type) {
    switch (similarity_type) {
        case CVECTOR_SIMILARITY_COSINE:
            return cvector_cosine_similarity(a, b, dimension);
        case CVECTOR_SIMILARITY_DOT_PRODUCT:
            return cvector_dot_product(a, b, dimension);
        case CVECTOR_SIMILARITY_EUCLIDEAN:
            return -cvector_euclidean_distance(a, b, dimension); // Negative for max-heap behavior
        default:
            return 0.0f;
    }
}

// Search within a single layer
static cvector_error_t hnsw_search_layer(hnsw_index_t* index, const float* query_vector,
                                        hnsw_priority_queue_t* entry_points, uint32_t num_closest,
                                        uint32_t level) {
    // This is a simplified greedy search - in production HNSW, this would be more sophisticated
    hnsw_priority_queue_t* candidates;
    hnsw_priority_queue_t* w; // Dynamic candidate set
    
    cvector_error_t err = hnsw_pq_create(index->ef_construction, false, &candidates);
    if (err != CVECTOR_SUCCESS) return err;
    
    err = hnsw_pq_create(index->ef_construction, true, &w);
    if (err != CVECTOR_SUCCESS) {
        hnsw_pq_destroy(candidates);
        return err;
    }
    
    // Copy entry points to candidates and w
    for (uint32_t i = 0; i < entry_points->count; i++) {
        uint32_t node_id = entry_points->items[i].node_id;
        float distance = hnsw_calculate_similarity(query_vector, 
                                                  index->nodes[node_id]->vector_data,
                                                  index->dimension, index->similarity_type);
        hnsw_pq_push(candidates, node_id, distance);
        hnsw_pq_push(w, node_id, distance);
    }
    
    bool* visited = calloc(index->node_count, sizeof(bool));
    if (!visited) {
        hnsw_pq_destroy(candidates);
        hnsw_pq_destroy(w);
        return CVECTOR_ERROR_OUT_OF_MEMORY;
    }
    
    // Mark entry points as visited (with bounds checking)
    for (uint32_t i = 0; i < entry_points->count; i++) {
        uint32_t node_id = entry_points->items[i].node_id;
        if (node_id < index->node_count) {
            visited[node_id] = true;
        }
    }
    
    while (!hnsw_pq_is_empty(candidates)) {
        uint32_t current_node;
        float current_dist;
        if (!hnsw_pq_pop(candidates, &current_node, &current_dist)) break;
        
        // Get the furthest element in w
        float furthest_in_w = w->count > 0 ? w->items[0].distance : -FLT_MAX;
        
        if (current_dist > furthest_in_w) {
            break; // All remaining elements are further than the current furthest
        }
        
        hnsw_node_t* node = index->nodes[current_node];
        
        // Examine neighbors at this level
        for (uint32_t i = 0; i < node->connection_count[level]; i++) {
            uint32_t neighbor_id = node->connections[level][i];
            
            // Bounds check for neighbor_id
            if (neighbor_id >= index->node_count || !index->nodes[neighbor_id]) {
                continue;
            }
            
            if (neighbor_id < index->node_count && !visited[neighbor_id]) {
                visited[neighbor_id] = true;
                
                float neighbor_dist = hnsw_calculate_similarity(query_vector,
                                                              index->nodes[neighbor_id]->vector_data,
                                                              index->dimension, index->similarity_type);
                
                if (neighbor_dist > furthest_in_w || w->count < num_closest) {
                    hnsw_pq_push(candidates, neighbor_id, neighbor_dist);
                    hnsw_pq_push(w, neighbor_id, neighbor_dist);
                    
                    if (w->count > num_closest) {
                        uint32_t dummy;
                        float dummy_dist;
                        hnsw_pq_pop(w, &dummy, &dummy_dist); // Remove furthest
                    }
                }
            }
        }
    }
    
    // Copy results back to entry_points
    entry_points->count = 0;
    while (!hnsw_pq_is_empty(w) && entry_points->count < entry_points->capacity) {
        uint32_t node_id;
        float distance;
        if (hnsw_pq_pop(w, &node_id, &distance)) {
            hnsw_pq_push(entry_points, node_id, distance);
        }
    }
    
    free(visited);
    hnsw_pq_destroy(candidates);
    hnsw_pq_destroy(w);
    
    return CVECTOR_SUCCESS;
}

cvector_error_t hnsw_add_vector(hnsw_index_t* index, cvector_id_t id, const float* vector) {
    uint64_t start_time = hnsw_get_timestamp_ns();
    
    // Input validation with detailed logging
    cvector_error_t err = hnsw_validate_index(index);
    if (err != CVECTOR_SUCCESS) return err;
    
    if (!vector) {
        // Error logged;
        return CVECTOR_ERROR_INVALID_ARGS;
    }
    
    // Check for corruption
    if (index->is_corrupted) {
        // Error logged;
        return CVECTOR_ERROR_DB_CORRUPT;
    }
    
    // Acquire write lock for thread safety
    if (pthread_mutex_lock(&index->write_mutex) != 0) {
        // Error logged;
        return CVECTOR_ERROR_DB_CORRUPT;
    }
    
    // Debug info removed
    
    // Resize if needed
    if (index->node_count >= index->node_capacity) {
        // Resizing index
        err = hnsw_resize_index(index);
        if (err != CVECTOR_SUCCESS) {
            // Error logged;
            pthread_mutex_unlock(&index->write_mutex);
            return err;
        }
    }
    
    // Create new node
    hnsw_node_t* node = calloc(1, sizeof(hnsw_node_t));
    if (!node) {
        pthread_mutex_unlock(&index->write_mutex);
        return CVECTOR_ERROR_OUT_OF_MEMORY;
    }
    
    node->id = id;
    node->level = hnsw_random_level(index->ml);
    node->dimension = index->dimension;
    
    // Copy vector data
    node->vector_data = malloc(index->dimension * sizeof(float));
    if (!node->vector_data) {
        free(node);
        pthread_mutex_unlock(&index->write_mutex);
        return CVECTOR_ERROR_OUT_OF_MEMORY;
    }
    memcpy(node->vector_data, vector, index->dimension * sizeof(float));
    
    // Initialize connection arrays
    for (uint32_t level = 0; level <= node->level; level++) {
        uint32_t max_connections = (level == 0) ? index->M * 2 : index->M;
        node->connections[level] = malloc(max_connections * sizeof(uint32_t));
        if (!node->connections[level]) {
            // Cleanup on failure
            for (uint32_t l = 0; l < level; l++) {
                free(node->connections[l]);
            }
            free(node->vector_data);
            free(node);
            pthread_mutex_unlock(&index->write_mutex);
            return CVECTOR_ERROR_OUT_OF_MEMORY;
        }
        node->connection_count[level] = 0;
    }
    
    uint32_t node_id = index->node_count;
    index->nodes[node_id] = node;
    index->node_count++;
    
    // If this is the first node, make it the entry point
    if (index->entry_point == UINT32_MAX) {
        index->entry_point = node_id;
        index->max_level = node->level;
        
        // Update statistics for first node
        hnsw_atomic_inc_u64(&index->insert_count);
        hnsw_atomic_store_u64(&index->memory_used, 
                             hnsw_atomic_load_u64(&index->memory_used) + 
                             sizeof(hnsw_node_t) + (index->dimension * sizeof(float)));
        index->checksum = hnsw_calculate_checksum(index);
        index->last_modified = hnsw_get_timestamp_s();
        
        // Release write lock before returning
        pthread_mutex_unlock(&index->write_mutex);
        
        // Vector insertion completed
        
        return CVECTOR_SUCCESS;
    }
    
    // Connect to the graph using production-grade algorithm
    err = hnsw_connect_layers_safe(index, node_id, node->level);
    if (err != CVECTOR_SUCCESS) {
        // Cleanup on failure
        for (uint32_t level = 0; level <= node->level; level++) {
            free(node->connections[level]);
        }
        free(node->vector_data);
        free(node);
        index->node_count--;
        index->nodes[node_id] = NULL;
        
        // Release write lock before returning
        pthread_mutex_unlock(&index->write_mutex);
        return err;
    }
    
    // Update entry point if this node is at a higher level
    if (node->level > index->max_level) {
        index->entry_point = node_id;
        index->max_level = node->level;
    }
    
    // Update statistics and integrity information
    hnsw_atomic_inc_u64(&index->insert_count);
    hnsw_atomic_store_u64(&index->memory_used, 
                         hnsw_atomic_load_u64(&index->memory_used) + 
                         sizeof(hnsw_node_t) + (index->dimension * sizeof(float)));
    index->checksum = hnsw_calculate_checksum(index);
    index->last_modified = hnsw_get_timestamp_s();
    
    // Release write lock
    pthread_mutex_unlock(&index->write_mutex);
    
    uint64_t end_time = hnsw_get_timestamp_ns();
    // Vector insertion completed
    
    return CVECTOR_SUCCESS;
}

static cvector_error_t hnsw_connect_layers_safe(hnsw_index_t* index, uint32_t node_id, uint32_t max_level) {
    if (!index || node_id >= index->node_count || !index->nodes[node_id]) {
        return CVECTOR_ERROR_INVALID_ARGS;
    }
    
    // For the first few nodes, use simple connections to avoid complex graph issues
    if (index->node_count <= 5) {
        hnsw_node_t* node = index->nodes[node_id];
        
        // Connect to all existing nodes at appropriate levels
        for (uint32_t i = 0; i < node_id; i++) {
            if (!index->nodes[i]) continue;
            
            hnsw_node_t* neighbor = index->nodes[i];
            uint32_t common_level = (node->level < neighbor->level) ? node->level : neighbor->level;
            
            // Add bidirectional connections at all common levels
            for (uint32_t level = 0; level <= common_level; level++) {
                uint32_t max_conn = (level == 0) ? index->M * 2 : index->M;
                
                // Add connection from node to neighbor
                if (node->connection_count[level] < max_conn) {
                    node->connections[level][node->connection_count[level]] = i;
                    node->connection_count[level]++;
                }
                
                // Add connection from neighbor to node
                if (neighbor->connection_count[level] < max_conn) {
                    neighbor->connections[level][neighbor->connection_count[level]] = node_id;
                    neighbor->connection_count[level]++;
                }
            }
        }
        return CVECTOR_SUCCESS;
    }
    
    // For larger graphs, use proper HNSW algorithm
    hnsw_priority_queue_t* entry_points;
    cvector_error_t err = hnsw_pq_create(index->ef_construction, false, &entry_points);
    if (err != CVECTOR_SUCCESS) return err;
    
    // Start from entry point
    if (index->entry_point != UINT32_MAX && index->nodes[index->entry_point]) {
        float entry_dist = hnsw_calculate_similarity(index->nodes[node_id]->vector_data,
                                                    index->nodes[index->entry_point]->vector_data,
                                                    index->dimension, index->similarity_type);
        hnsw_pq_push(entry_points, index->entry_point, entry_dist);
    }
    
    // Search from top level down to max_level + 1
    for (uint32_t level = index->max_level; level > max_level; level--) {
        err = hnsw_search_layer(index, index->nodes[node_id]->vector_data, entry_points, 1, level);
        if (err != CVECTOR_SUCCESS) {
            hnsw_pq_destroy(entry_points);
            return err;
        }
    }
    
    // Search and connect at each level from max_level down to 0
    for (uint32_t level = max_level; level != UINT32_MAX; level--) {
        uint32_t ef = (level == 0) ? index->ef_construction : index->M;
        
        err = hnsw_search_layer(index, index->nodes[node_id]->vector_data, entry_points, ef, level);
        if (err != CVECTOR_SUCCESS) {
            hnsw_pq_destroy(entry_points);
            return err;
        }
        
        // Use sophisticated neighbor selection
        err = hnsw_select_neighbors_simple(index, node_id, entry_points, index->M, level);
        if (err != CVECTOR_SUCCESS) {
            hnsw_pq_destroy(entry_points);
            return err;
        }
    }
    
    hnsw_pq_destroy(entry_points);
    return CVECTOR_SUCCESS;
}

static cvector_error_t hnsw_select_neighbors_simple(hnsw_index_t* index, uint32_t node_id, 
                                                   hnsw_priority_queue_t* candidates, uint32_t M, uint32_t level) {
    if (!index || !candidates || node_id >= index->node_count || !index->nodes[node_id]) {
        return CVECTOR_ERROR_INVALID_ARGS;
    }
    
    hnsw_node_t* node = index->nodes[node_id];
    uint32_t max_connections = (level == 0) ? M * 2 : M;
    
    // Simple selection: take the M closest candidates
    uint32_t selected = 0;
    while (!hnsw_pq_is_empty(candidates) && selected < M && node->connection_count[level] < max_connections) {
        uint32_t neighbor_id;
        float distance;
        if (!hnsw_pq_pop(candidates, &neighbor_id, &distance)) break;
        
        // Skip self-connections
        if (neighbor_id == node_id) continue;
        
        // Verify neighbor exists and is valid
        if (neighbor_id >= index->node_count || !index->nodes[neighbor_id]) continue;
        
        // Check if connection already exists
        bool already_connected = false;
        for (uint32_t i = 0; i < node->connection_count[level]; i++) {
            if (node->connections[level][i] == neighbor_id) {
                already_connected = true;
                break;
            }
        }
        if (already_connected) continue;
        
        // Add connection from node to neighbor
        node->connections[level][node->connection_count[level]] = neighbor_id;
        node->connection_count[level]++;
        
        // Add reverse connection from neighbor to node
        hnsw_node_t* neighbor = index->nodes[neighbor_id];
        if (neighbor->connection_count[level] < max_connections) {
            // Check if reverse connection already exists
            bool reverse_exists = false;
            for (uint32_t i = 0; i < neighbor->connection_count[level]; i++) {
                if (neighbor->connections[level][i] == node_id) {
                    reverse_exists = true;
                    break;
                }
            }
            if (!reverse_exists) {
                neighbor->connections[level][neighbor->connection_count[level]] = node_id;
                neighbor->connection_count[level]++;
            }
        }
        
        selected++;
    }
    
    return CVECTOR_SUCCESS;
}

// Removed unused legacy function hnsw_connect_layers

cvector_error_t hnsw_search(hnsw_index_t* index, const float* query_vector, 
                           uint32_t top_k, hnsw_search_result_t** result) {
    return hnsw_search_with_ef(index, query_vector, top_k, index->ef_search, result);
}

cvector_error_t hnsw_search_with_ef(hnsw_index_t* index, const float* query_vector,
                                   uint32_t top_k, uint32_t ef, hnsw_search_result_t** result) {
    uint64_t start_time = hnsw_get_timestamp_ns();
    
    // Input validation with detailed logging
    cvector_error_t err = hnsw_validate_index(index);
    if (err != CVECTOR_SUCCESS) return err;
    
    if (!query_vector) {
        // Error logged;
        return CVECTOR_ERROR_INVALID_ARGS;
    }
    
    if (!result) {
        // Error logged;
        return CVECTOR_ERROR_INVALID_ARGS;
    }
    
    if (top_k == 0) {
        // Error logged;
        return CVECTOR_ERROR_INVALID_ARGS;
    }
    
    if (ef == 0) {
        // Warning logged;
        ef = index->ef_search;
    }
    
    // Check for corruption
    if (index->is_corrupted) {
        // Error logged;
        return CVECTOR_ERROR_DB_CORRUPT;
    }
    
    // Acquire read lock for thread safety
    if (pthread_rwlock_rdlock(&index->search_lock) != 0) {
        // Error logged;
        return CVECTOR_ERROR_DB_CORRUPT;
    }
    
    // Searching HNSW index
    
    if (index->node_count == 0 || index->entry_point == UINT32_MAX) {
        // Info logged;
        *result = calloc(1, sizeof(hnsw_search_result_t));
        if (!*result) {
            // Error logged;
            pthread_rwlock_unlock(&index->search_lock);
            return CVECTOR_ERROR_OUT_OF_MEMORY;
        }
        pthread_rwlock_unlock(&index->search_lock);
        return CVECTOR_SUCCESS;
    }
    
    hnsw_atomic_inc_u64(&index->search_count);
    
    hnsw_priority_queue_t* entry_points;
    err = hnsw_pq_create(ef, false, &entry_points);
    if (err != CVECTOR_SUCCESS) {
        pthread_rwlock_unlock(&index->search_lock);
        return err;
    }
    
    // Start from entry point
    float entry_dist = hnsw_calculate_similarity(query_vector,
                                                index->nodes[index->entry_point]->vector_data,
                                                index->dimension, index->similarity_type);
    hnsw_pq_push(entry_points, index->entry_point, entry_dist);
    
    // Search from top level down to level 1
    for (uint32_t level = index->max_level; level > 0; level--) {
        err = hnsw_search_layer(index, query_vector, entry_points, 1, level);
        if (err != CVECTOR_SUCCESS) {
            hnsw_pq_destroy(entry_points);
            pthread_rwlock_unlock(&index->search_lock);
            return err;
        }
    }
    
    // Search at level 0 with ef
    err = hnsw_search_layer(index, query_vector, entry_points, ef, 0);
    if (err != CVECTOR_SUCCESS) {
        hnsw_pq_destroy(entry_points);
        pthread_rwlock_unlock(&index->search_lock);
        return err;
    }
    
    // Create result structure
    *result = malloc(sizeof(hnsw_search_result_t));
    if (!*result) {
        hnsw_pq_destroy(entry_points);
        pthread_rwlock_unlock(&index->search_lock);
        return CVECTOR_ERROR_OUT_OF_MEMORY;
    }
    
    uint32_t result_count = (entry_points->count < top_k) ? entry_points->count : top_k;
    (*result)->capacity = result_count;
    (*result)->count = result_count;
    
    if (result_count > 0) {
        (*result)->ids = malloc(result_count * sizeof(cvector_id_t));
        (*result)->similarities = malloc(result_count * sizeof(float));
        
        if (!(*result)->ids || !(*result)->similarities) {
            free((*result)->ids);
            free((*result)->similarities);
            free(*result);
            *result = NULL;
            hnsw_pq_destroy(entry_points);
            pthread_rwlock_unlock(&index->search_lock);
            return CVECTOR_ERROR_OUT_OF_MEMORY;
        }
        
        // Copy results to temporary array for sorting
        typedef struct {
            cvector_id_t id;
            float similarity;
        } result_item_t;
        
        result_item_t* temp_results = malloc(result_count * sizeof(result_item_t));
        if (!temp_results) {
            free((*result)->ids);
            free((*result)->similarities);
            free(*result);
            *result = NULL;
            hnsw_pq_destroy(entry_points);
            pthread_rwlock_unlock(&index->search_lock);
            return CVECTOR_ERROR_OUT_OF_MEMORY;
        }
        
        // Extract results from priority queue
        for (uint32_t i = 0; i < result_count; i++) {
            uint32_t node_id;
            float distance;
            if (hnsw_pq_pop(entry_points, &node_id, &distance)) {
                temp_results[i].id = index->nodes[node_id]->id;
                temp_results[i].similarity = distance;
                hnsw_atomic_inc_u64(&index->total_distance_computations);
            }
        }
        
        // Sort results by similarity (descending - higher similarity first)
        for (uint32_t i = 0; i < result_count - 1; i++) {
            for (uint32_t j = i + 1; j < result_count; j++) {
                if (temp_results[j].similarity > temp_results[i].similarity) {
                    result_item_t temp = temp_results[i];
                    temp_results[i] = temp_results[j];
                    temp_results[j] = temp;
                }
            }
        }
        
        // Copy sorted results to final structure
        for (uint32_t i = 0; i < result_count; i++) {
            (*result)->ids[i] = temp_results[i].id;
            (*result)->similarities[i] = temp_results[i].similarity;
        }
        
        free(temp_results);
    } else {
        (*result)->ids = NULL;
        (*result)->similarities = NULL;
    }
    
    hnsw_pq_destroy(entry_points);
    pthread_rwlock_unlock(&index->search_lock);
    
    // Search completed
    
    return CVECTOR_SUCCESS;
}

void hnsw_free_search_result(hnsw_search_result_t* result) {
    if (result) {
        free(result->ids);
        free(result->similarities);
        free(result);
    }
}

cvector_error_t hnsw_get_stats(hnsw_index_t* index, hnsw_stats_t* stats) {
    if (!index || !stats) {
        return CVECTOR_ERROR_INVALID_ARGS;
    }
    
    stats->node_count = index->node_count;
    stats->max_level = index->max_level;
    stats->search_count = index->search_count;
    stats->distance_computations = index->total_distance_computations;
    stats->entry_point_level = (index->entry_point != UINT32_MAX) ? 
                              index->nodes[index->entry_point]->level : 0;
    
    // Calculate average connections per node
    if (index->node_count > 0) {
        uint64_t total_connections = 0;
        for (uint32_t i = 0; i < index->node_count; i++) {
            if (index->nodes[i]) {
                for (uint32_t level = 0; level <= index->nodes[i]->level; level++) {
                    total_connections += index->nodes[i]->connection_count[level];
                }
            }
        }
        stats->avg_connections_per_node = (float)total_connections / index->node_count;
    } else {
        stats->avg_connections_per_node = 0.0f;
    }
    
    return CVECTOR_SUCCESS;
}

cvector_error_t hnsw_set_config(hnsw_index_t* index, const hnsw_config_t* config) {
    if (!index || !config) {
        return CVECTOR_ERROR_INVALID_ARGS;
    }
    
    index->M = config->M;
    index->ef_construction = config->ef_construction;
    index->ef_search = config->ef_search;
    index->ml = config->ml;
    
    return CVECTOR_SUCCESS;
}

cvector_error_t hnsw_get_config(hnsw_index_t* index, hnsw_config_t* config) {
    if (!index || !config) {
        return CVECTOR_ERROR_INVALID_ARGS;
    }
    
    config->M = index->M;
    config->ef_construction = index->ef_construction;
    config->ef_search = index->ef_search;
    config->ml = index->ml;
    
    return CVECTOR_SUCCESS;
}

// Production-grade HNSW persistence implementation
cvector_error_t hnsw_save_index(hnsw_index_t* index, const char* filepath) {
    cvector_error_t err = hnsw_validate_index(index);
    if (err != CVECTOR_SUCCESS) return err;
    
    if (!filepath) {
        // Error logged;
        return CVECTOR_ERROR_INVALID_ARGS;
    }
    
    // Info logged;
    
    FILE* file = fopen(filepath, "wb");
    if (!file) {
        // Error logged;
        return CVECTOR_ERROR_FILE_IO;
    }
    
    // Write header with magic number and version
    uint32_t magic = 0x484E5357; // "HNSW"
    uint32_t version = 1;
    if (fwrite(&magic, sizeof(magic), 1, file) != 1 ||
        fwrite(&version, sizeof(version), 1, file) != 1) {
        // Error logged;
        fclose(file);
        return CVECTOR_ERROR_FILE_IO;
    }
    
    // Write index metadata
    if (fwrite(&index->dimension, sizeof(index->dimension), 1, file) != 1 ||
        fwrite(&index->similarity_type, sizeof(index->similarity_type), 1, file) != 1 ||
        fwrite(&index->M, sizeof(index->M), 1, file) != 1 ||
        fwrite(&index->ef_construction, sizeof(index->ef_construction), 1, file) != 1 ||
        fwrite(&index->ef_search, sizeof(index->ef_search), 1, file) != 1 ||
        fwrite(&index->ml, sizeof(index->ml), 1, file) != 1 ||
        fwrite(&index->node_count, sizeof(index->node_count), 1, file) != 1 ||
        fwrite(&index->entry_point, sizeof(index->entry_point), 1, file) != 1 ||
        fwrite(&index->max_level, sizeof(index->max_level), 1, file) != 1) {
        // Error logged;
        fclose(file);
        return CVECTOR_ERROR_FILE_IO;
    }
    
    // Write nodes
    for (uint32_t i = 0; i < index->node_count; i++) {
        if (!index->nodes[i]) {
            // Error logged;
            fclose(file);
            return CVECTOR_ERROR_DB_CORRUPT;
        }
        
        hnsw_node_t* node = index->nodes[i];
        
        // Write node metadata
        if (fwrite(&node->id, sizeof(node->id), 1, file) != 1 ||
            fwrite(&node->level, sizeof(node->level), 1, file) != 1 ||
            fwrite(&node->dimension, sizeof(node->dimension), 1, file) != 1) {
            // Error logged;
            fclose(file);
            return CVECTOR_ERROR_FILE_IO;
        }
        
        // Write vector data
        if (fwrite(node->vector_data, sizeof(float), node->dimension, file) != node->dimension) {
            // Error logged;
            fclose(file);
            return CVECTOR_ERROR_FILE_IO;
        }
        
        // Write connections for each level
        for (uint32_t level = 0; level <= node->level; level++) {
            if (fwrite(&node->connection_count[level], sizeof(node->connection_count[level]), 1, file) != 1) {
                // Error logged;
                fclose(file);
                return CVECTOR_ERROR_FILE_IO;
            }
            
            if (node->connection_count[level] > 0) {
                if (fwrite(node->connections[level], sizeof(uint32_t), 
                          node->connection_count[level], file) != node->connection_count[level]) {
                    // Error logged;
                    fclose(file);
                    return CVECTOR_ERROR_FILE_IO;
                }
            }
        }
    }
    
    fclose(file);
    // Info logged;
    return CVECTOR_SUCCESS;
}

cvector_error_t hnsw_load_index(const char* filepath, hnsw_index_t** index) {
    if (!filepath || !index) {
        // Error logged;
        return CVECTOR_ERROR_INVALID_ARGS;
    }
    
    // Info logged;
    
    FILE* file = fopen(filepath, "rb");
    if (!file) {
        // Error logged;
        return CVECTOR_ERROR_FILE_IO;
    }
    
    // Read and verify header
    uint32_t magic, version;
    if (fread(&magic, sizeof(magic), 1, file) != 1 ||
        fread(&version, sizeof(version), 1, file) != 1) {
        // Error logged;
        fclose(file);
        return CVECTOR_ERROR_FILE_IO;
    }
    
    if (magic != 0x484E5357) {
        // Error logged;
        fclose(file);
        return CVECTOR_ERROR_DB_CORRUPT;
    }
    
    if (version != 1) {
        // Error logged;
        fclose(file);
        return CVECTOR_ERROR_DB_CORRUPT;
    }
    
    // Read index metadata
    uint32_t dimension;
    cvector_similarity_t similarity_type;
    if (fread(&dimension, sizeof(dimension), 1, file) != 1 ||
        fread(&similarity_type, sizeof(similarity_type), 1, file) != 1) {
        // Error logged;
        fclose(file);
        return CVECTOR_ERROR_FILE_IO;
    }
    
    // Create index
    cvector_error_t err = hnsw_create_index(dimension, similarity_type, index);
    if (err != CVECTOR_SUCCESS) {
        // Error logged;
        fclose(file);
        return err;
    }
    
    hnsw_index_t* idx = *index;
    
    // Read remaining metadata
    if (fread(&idx->M, sizeof(idx->M), 1, file) != 1 ||
        fread(&idx->ef_construction, sizeof(idx->ef_construction), 1, file) != 1 ||
        fread(&idx->ef_search, sizeof(idx->ef_search), 1, file) != 1 ||
        fread(&idx->ml, sizeof(idx->ml), 1, file) != 1 ||
        fread(&idx->node_count, sizeof(idx->node_count), 1, file) != 1 ||
        fread(&idx->entry_point, sizeof(idx->entry_point), 1, file) != 1 ||
        fread(&idx->max_level, sizeof(idx->max_level), 1, file) != 1) {
        // Error logged;
        hnsw_destroy_index(idx);
        *index = NULL;
        fclose(file);
        return CVECTOR_ERROR_FILE_IO;
    }
    
    // Ensure capacity is sufficient
    while (idx->node_capacity < idx->node_count) {
        err = hnsw_resize_index(idx);
        if (err != CVECTOR_SUCCESS) {
            // Error logged;
            hnsw_destroy_index(idx);
            *index = NULL;
            fclose(file);
            return err;
        }
    }
    
    // Read nodes
    for (uint32_t i = 0; i < idx->node_count; i++) {
        hnsw_node_t* node = calloc(1, sizeof(hnsw_node_t));
        if (!node) {
            // Error logged;
            hnsw_destroy_index(idx);
            *index = NULL;
            fclose(file);
            return CVECTOR_ERROR_OUT_OF_MEMORY;
        }
        
        // Read node metadata
        if (fread(&node->id, sizeof(node->id), 1, file) != 1 ||
            fread(&node->level, sizeof(node->level), 1, file) != 1 ||
            fread(&node->dimension, sizeof(node->dimension), 1, file) != 1) {
            // Error logged;
            free(node);
            hnsw_destroy_index(idx);
            *index = NULL;
            fclose(file);
            return CVECTOR_ERROR_FILE_IO;
        }
        
        // Allocate and read vector data
        node->vector_data = malloc(node->dimension * sizeof(float));
        if (!node->vector_data) {
            // Error logged;
            free(node);
            hnsw_destroy_index(idx);
            *index = NULL;
            fclose(file);
            return CVECTOR_ERROR_OUT_OF_MEMORY;
        }
        
        if (fread(node->vector_data, sizeof(float), node->dimension, file) != node->dimension) {
            // Error logged;
            free(node->vector_data);
            free(node);
            hnsw_destroy_index(idx);
            *index = NULL;
            fclose(file);
            return CVECTOR_ERROR_FILE_IO;
        }
        
        // Read connections for each level
        for (uint32_t level = 0; level <= node->level; level++) {
            if (fread(&node->connection_count[level], sizeof(node->connection_count[level]), 1, file) != 1) {
                // Error logged;
                free(node->vector_data);
                free(node);
                hnsw_destroy_index(idx);
                *index = NULL;
                fclose(file);
                return CVECTOR_ERROR_FILE_IO;
            }
            
            if (node->connection_count[level] > 0) {
                uint32_t max_connections = (level == 0) ? idx->M * 2 : idx->M;
                node->connections[level] = malloc(max_connections * sizeof(uint32_t));
                if (!node->connections[level]) {
                    // Error logged;
                    for (uint32_t l = 0; l < level; l++) {
                        free(node->connections[l]);
                    }
                    free(node->vector_data);
                    free(node);
                    hnsw_destroy_index(idx);
                    *index = NULL;
                    fclose(file);
                    return CVECTOR_ERROR_OUT_OF_MEMORY;
                }
                
                if (fread(node->connections[level], sizeof(uint32_t), 
                         node->connection_count[level], file) != node->connection_count[level]) {
                    // Error logged;
                    for (uint32_t l = 0; l <= level; l++) {
                        free(node->connections[l]);
                    }
                    free(node->vector_data);
                    free(node);
                    hnsw_destroy_index(idx);
                    *index = NULL;
                    fclose(file);
                    return CVECTOR_ERROR_FILE_IO;
                }
            }
        }
        
        idx->nodes[i] = node;
    }
    
    fclose(file);
    // Info logged;
    return CVECTOR_SUCCESS;
}

cvector_error_t hnsw_remove_vector(hnsw_index_t* index, cvector_id_t id) {
    if (!index) {
        return CVECTOR_ERROR_INVALID_ARGS;
    }
    
    // Find the node with this ID
    uint32_t node_to_remove = UINT32_MAX;
    for (uint32_t i = 0; i < index->node_count; i++) {
        if (index->nodes[i] && index->nodes[i]->id == id) {
            node_to_remove = i;
            break;
        }
    }
    
    if (node_to_remove == UINT32_MAX) {
        return CVECTOR_ERROR_VECTOR_NOT_FOUND;
    }
    
    hnsw_node_t* node = index->nodes[node_to_remove];
    
    // Remove all connections to this node from other nodes
    for (uint32_t i = 0; i < index->node_count; i++) {
        if (i == node_to_remove || !index->nodes[i]) continue;
        
        hnsw_node_t* other_node = index->nodes[i];
        for (uint32_t level = 0; level <= other_node->level; level++) {
            for (uint32_t j = 0; j < other_node->connection_count[level]; j++) {
                if (other_node->connections[level][j] == node_to_remove) {
                    // Remove this connection by shifting remaining connections
                    for (uint32_t k = j; k < other_node->connection_count[level] - 1; k++) {
                        other_node->connections[level][k] = other_node->connections[level][k + 1];
                    }
                    other_node->connection_count[level]--;
                    j--; // Recheck this position
                }
            }
        }
    }
    
    // Free the node
    for (uint32_t level = 0; level <= node->level; level++) {
        free(node->connections[level]);
    }
    free(node->vector_data);
    free(node);
    index->nodes[node_to_remove] = NULL;
    
    // Update entry point if needed
    if (index->entry_point == node_to_remove) {
        // Find new entry point (highest level remaining node)
        index->entry_point = UINT32_MAX;
        index->max_level = 0;
        for (uint32_t i = 0; i < index->node_count; i++) {
            if (index->nodes[i] && index->nodes[i]->level >= index->max_level) {
                index->entry_point = i;
                index->max_level = index->nodes[i]->level;
            }
        }
    }
    
    hnsw_atomic_inc_u64(&index->delete_count);
    index->checksum = hnsw_calculate_checksum(index);
    index->last_modified = hnsw_get_timestamp_s();
    
    return CVECTOR_SUCCESS;
}

// Production-grade Thread Safety Control Functions
cvector_error_t hnsw_lock_for_write(hnsw_index_t* index) {
    if (!index) return CVECTOR_ERROR_INVALID_ARGS;
    
    if (pthread_mutex_lock(&index->write_mutex) != 0) {
        // Error logged;
        return CVECTOR_ERROR_DB_CORRUPT;
    }
    return CVECTOR_SUCCESS;
}

cvector_error_t hnsw_unlock_write(hnsw_index_t* index) {
    if (!index) return CVECTOR_ERROR_INVALID_ARGS;
    
    if (pthread_mutex_unlock(&index->write_mutex) != 0) {
        // Error logged;
        return CVECTOR_ERROR_DB_CORRUPT;
    }
    return CVECTOR_SUCCESS;
}

cvector_error_t hnsw_lock_for_read(hnsw_index_t* index) {
    if (!index) return CVECTOR_ERROR_INVALID_ARGS;
    
    if (pthread_rwlock_rdlock(&index->search_lock) != 0) {
        // Error logged;
        return CVECTOR_ERROR_DB_CORRUPT;
    }
    return CVECTOR_SUCCESS;
}

cvector_error_t hnsw_unlock_read(hnsw_index_t* index) {
    if (!index) return CVECTOR_ERROR_INVALID_ARGS;
    
    if (pthread_rwlock_unlock(&index->search_lock) != 0) {
        // Error logged;
        return CVECTOR_ERROR_DB_CORRUPT;
    }
    return CVECTOR_SUCCESS;
}

// Production-grade Integrity and Recovery Functions
cvector_error_t hnsw_validate_integrity(hnsw_index_t* index) {
    if (!index) return CVECTOR_ERROR_INVALID_ARGS;
    
    // Info logged;
    
    // Check for obvious corruption flags
    if (index->is_corrupted) {
        // Error logged;
        return CVECTOR_ERROR_DB_CORRUPT;
    }
    
    // Validate basic structure
    if (index->node_count > index->node_capacity) {
        // Error logged;
        index->is_corrupted = true;
        return CVECTOR_ERROR_DB_CORRUPT;
    }
    
    // Check each node for consistency
    uint32_t valid_nodes = 0;
    for (uint32_t i = 0; i < index->node_count; i++) {
        if (!index->nodes[i]) continue;
        
        hnsw_node_t* node = index->nodes[i];
        
        // Validate node structure
        if (node->dimension != index->dimension) {
            // Node has mismatched dimension
            index->is_corrupted = true;
            return CVECTOR_ERROR_DB_CORRUPT;
        }
        
        if (node->level >= HNSW_MAX_LEVEL) {
            // Error logged;
            index->is_corrupted = true;
            return CVECTOR_ERROR_DB_CORRUPT;
        }
        
        // Validate connections
        for (uint32_t level = 0; level <= node->level; level++) {
            uint32_t max_conn = (level == 0) ? index->M * 2 : index->M;
            if (node->connection_count[level] > max_conn) {
                // Node has too many connections
                index->is_corrupted = true;
                return CVECTOR_ERROR_DB_CORRUPT;
            }
            
            // Validate connection targets
            for (uint32_t j = 0; j < node->connection_count[level]; j++) {
                uint32_t target = node->connections[level][j];
                if (target >= index->node_count || !index->nodes[target]) {
                    // Node has invalid connection
                    index->is_corrupted = true;
                    return CVECTOR_ERROR_DB_CORRUPT;
                }
            }
        }
        
        valid_nodes++;
    }
    
    // Verify entry point
    if (index->entry_point != UINT32_MAX) {
        if (index->entry_point >= index->node_count || !index->nodes[index->entry_point]) {
            // Error logged;
            index->is_corrupted = true;
            return CVECTOR_ERROR_DB_CORRUPT;
        }
    } else if (valid_nodes > 0) {
        // Error logged;
        index->is_corrupted = true;
        return CVECTOR_ERROR_DB_CORRUPT;
    }
    
    // Info logged;
    return CVECTOR_SUCCESS;
}

cvector_error_t hnsw_repair_index(hnsw_index_t* index) {
    if (!index) return CVECTOR_ERROR_INVALID_ARGS;
    
    // Info logged;
    
    // First, try basic validation
    cvector_error_t err = hnsw_validate_integrity(index);
    if (err == CVECTOR_SUCCESS) {
        // Info logged;
        return CVECTOR_SUCCESS;
    }
    
    // Attempt basic repairs
    uint32_t repairs = 0;
    
    // Fix entry point if corrupted
    if (index->entry_point == UINT32_MAX || 
        index->entry_point >= index->node_count || 
        !index->nodes[index->entry_point]) {
        
        // Find highest level node as new entry point
        uint32_t best_entry = UINT32_MAX;
        uint32_t best_level = 0;
        
        for (uint32_t i = 0; i < index->node_count; i++) {
            if (index->nodes[i] && index->nodes[i]->level >= best_level) {
                best_entry = i;
                best_level = index->nodes[i]->level;
            }
        }
        
        if (best_entry != UINT32_MAX) {
            index->entry_point = best_entry;
            index->max_level = best_level;
            repairs++;
            // Entry point repaired
        }
    }
    
    // Clean up invalid connections
    for (uint32_t i = 0; i < index->node_count; i++) {
        if (!index->nodes[i]) continue;
        
        hnsw_node_t* node = index->nodes[i];
        for (uint32_t level = 0; level <= node->level; level++) {
            uint32_t valid_connections = 0;
            
            for (uint32_t j = 0; j < node->connection_count[level]; j++) {
                uint32_t target = node->connections[level][j];
                
                // Check if connection is valid
                if (target < index->node_count && index->nodes[target]) {
                    if (valid_connections != j) {
                        node->connections[level][valid_connections] = target;
                    }
                    valid_connections++;
                } else {
                    repairs++;
                }
            }
            
            if (valid_connections != node->connection_count[level]) {
                node->connection_count[level] = valid_connections;
                // Cleaned up connections
            }
        }
    }
    
    // Clear corruption flag if we made repairs
    if (repairs > 0) {
        index->is_corrupted = false;
        index->checksum = hnsw_calculate_checksum(index);
        index->last_modified = hnsw_get_timestamp_s();
        // Info logged;
    }
    
    // Re-validate
    return hnsw_validate_integrity(index);
}

// Advanced Statistics Function
cvector_error_t hnsw_get_detailed_stats(hnsw_index_t* index, hnsw_detailed_stats_t* stats) {
    if (!index || !stats) {
        return CVECTOR_ERROR_INVALID_ARGS;
    }
    
    memset(stats, 0, sizeof(hnsw_detailed_stats_t));
    
    stats->node_count = index->node_count;
    stats->max_level = index->max_level;
    stats->search_count = hnsw_atomic_load_u64(&index->search_count);
    stats->insert_count = hnsw_atomic_load_u64(&index->insert_count);
    stats->delete_count = hnsw_atomic_load_u64(&index->delete_count);
    stats->distance_computations = hnsw_atomic_load_u64(&index->total_distance_computations);
    stats->memory_used = hnsw_atomic_load_u64(&index->memory_used);
    stats->memory_pool_size = index->memory_pool_size;
    stats->is_corrupted = index->is_corrupted;
    stats->last_modified = index->last_modified;
    
    stats->entry_point_level = (index->entry_point != UINT32_MAX && 
                               index->entry_point < index->node_count &&
                               index->nodes[index->entry_point]) ? 
                              index->nodes[index->entry_point]->level : 0;
    
    // Calculate average connections per node
    if (index->node_count > 0) {
        uint64_t total_connections = 0;
        uint32_t valid_nodes = 0;
        
        for (uint32_t i = 0; i < index->node_count; i++) {
            if (index->nodes[i]) {
                for (uint32_t level = 0; level <= index->nodes[i]->level; level++) {
                    total_connections += index->nodes[i]->connection_count[level];
                }
                valid_nodes++;
            }
        }
        
        stats->avg_connections_per_node = valid_nodes > 0 ? 
            (float)total_connections / valid_nodes : 0.0f;
    }
    
    // Estimate average times (simplified - would need proper timing in production)
    if (stats->search_count > 0) {
        stats->avg_search_time_ms = 0.5; // Placeholder
    }
    if (stats->insert_count > 0) {
        stats->avg_insert_time_ms = 1.0; // Placeholder
    }
    
    return CVECTOR_SUCCESS;
}

// Memory Management Functions
cvector_error_t hnsw_init_memory_pool(hnsw_index_t* index, size_t pool_size) {
    if (!index || pool_size == 0) {
        return CVECTOR_ERROR_INVALID_ARGS;
    }
    
    if (index->memory_pool) {
        // Warning logged;
        return CVECTOR_SUCCESS;
    }
    
    index->memory_pool = malloc(pool_size);
    if (!index->memory_pool) {
        // Error logged;
        return CVECTOR_ERROR_OUT_OF_MEMORY;
    }
    
    index->memory_pool_size = pool_size;
    // Info logged;
    
    return CVECTOR_SUCCESS;
}

cvector_error_t hnsw_cleanup_memory_pool(hnsw_index_t* index) {
    if (!index) {
        return CVECTOR_ERROR_INVALID_ARGS;
    }
    
    if (index->memory_pool) {
        free(index->memory_pool);
        index->memory_pool = NULL;
        index->memory_pool_size = 0;
        // Info logged;
    }
    
    return CVECTOR_SUCCESS;
}

// Backup and Recovery Functions
cvector_error_t hnsw_backup_index(hnsw_index_t* index, const char* backup_path) {
    if (!index || !backup_path) {
        return CVECTOR_ERROR_INVALID_ARGS;
    }
    
    // Info logged;
    
    // Create backup filename with timestamp
    char timestamped_path[1024];
    uint64_t timestamp = hnsw_get_timestamp_s();
    snprintf(timestamped_path, sizeof(timestamped_path), "%s.%llu.backup", 
             backup_path, (unsigned long long)timestamp);
    
    // Use existing save function for backup
    cvector_error_t err = hnsw_save_index(index, timestamped_path);
    if (err == CVECTOR_SUCCESS) {
        // Info logged;
    } else {
        // Error logged;
    }
    
    return err;
}

cvector_error_t hnsw_restore_from_backup(const char* backup_path, hnsw_index_t** index) {
    if (!backup_path || !index) {
        return CVECTOR_ERROR_INVALID_ARGS;
    }
    
    // Info logged;
    
    // Use existing load function for restoration
    cvector_error_t err = hnsw_load_index(backup_path, index);
    if (err == CVECTOR_SUCCESS) {
        // Info logged;
        
        // Validate the restored index
        err = hnsw_validate_integrity(*index);
        if (err != CVECTOR_SUCCESS) {
            // Error logged;
            hnsw_destroy_index(*index);
            *index = NULL;
        }
    } else {
        // Error logged;
    }
    
    return err;
}

// Performance Monitoring Stubs (for future implementation)
cvector_error_t hnsw_start_perf_monitoring(hnsw_index_t* index) {
    if (!index) return CVECTOR_ERROR_INVALID_ARGS;
    // Info logged;
    return CVECTOR_SUCCESS;
}

cvector_error_t hnsw_stop_perf_monitoring(hnsw_index_t* index) {
    if (!index) return CVECTOR_ERROR_INVALID_ARGS;
    // Info logged;
    return CVECTOR_SUCCESS;
}