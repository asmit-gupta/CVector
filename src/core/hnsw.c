#include "hnsw.h"
#include "similarity.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <assert.h>
#include <stdio.h>
#include <stdarg.h>

// Production-grade logging levels
typedef enum {
    HNSW_LOG_ERROR = 0,
    HNSW_LOG_WARN = 1,
    HNSW_LOG_INFO = 2,
    HNSW_LOG_DEBUG = 3
} hnsw_log_level_t;

static hnsw_log_level_t g_hnsw_log_level = HNSW_LOG_WARN;

// Production-grade logging function
static void hnsw_log(hnsw_log_level_t level, const char* format, ...) {
    if (level > g_hnsw_log_level) return;
    
    const char* level_str[] = {"ERROR", "WARN", "INFO", "DEBUG"};
    fprintf(stderr, "[HNSW %s] ", level_str[level]);
    
    va_list args;
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);
    
    fprintf(stderr, "\n");
}

// Input validation helper
static cvector_error_t hnsw_validate_index(hnsw_index_t* index) {
    if (!index) {
        hnsw_log(HNSW_LOG_ERROR, "Index is NULL");
        return CVECTOR_ERROR_INVALID_ARGS;
    }
    if (!index->nodes) {
        hnsw_log(HNSW_LOG_ERROR, "Index nodes array is NULL");
        return CVECTOR_ERROR_DB_CORRUPT;
    }
    if (index->dimension == 0) {
        hnsw_log(HNSW_LOG_ERROR, "Index dimension is 0");
        return CVECTOR_ERROR_INVALID_ARGS;
    }
    if (index->node_count > index->node_capacity) {
        hnsw_log(HNSW_LOG_ERROR, "Node count (%u) exceeds capacity (%u)", 
                index->node_count, index->node_capacity);
        return CVECTOR_ERROR_DB_CORRUPT;
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
static cvector_error_t hnsw_connect_layers(hnsw_index_t* index, uint32_t node_id, uint32_t level);
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
    
    return CVECTOR_SUCCESS;
}

cvector_error_t hnsw_destroy_index(hnsw_index_t* index) {
    if (!index) {
        return CVECTOR_SUCCESS;
    }
    
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
    // Input validation with detailed logging
    cvector_error_t err = hnsw_validate_index(index);
    if (err != CVECTOR_SUCCESS) return err;
    
    if (!vector) {
        hnsw_log(HNSW_LOG_ERROR, "Vector data is NULL");
        return CVECTOR_ERROR_INVALID_ARGS;
    }
    
    hnsw_log(HNSW_LOG_DEBUG, "Adding vector ID %llu to HNSW index (current size: %u)", 
             (unsigned long long)id, index->node_count);
    
    // Resize if needed
    if (index->node_count >= index->node_capacity) {
        hnsw_log(HNSW_LOG_DEBUG, "Resizing index from %u to %u capacity", 
                index->node_capacity, index->node_capacity * 2);
        err = hnsw_resize_index(index);
        if (err != CVECTOR_SUCCESS) {
            hnsw_log(HNSW_LOG_ERROR, "Failed to resize index: %d", err);
            return err;
        }
    }
    
    // Create new node
    hnsw_node_t* node = calloc(1, sizeof(hnsw_node_t));
    if (!node) {
        return CVECTOR_ERROR_OUT_OF_MEMORY;
    }
    
    node->id = id;
    node->level = hnsw_random_level(index->ml);
    node->dimension = index->dimension;
    
    // Copy vector data
    node->vector_data = malloc(index->dimension * sizeof(float));
    if (!node->vector_data) {
        free(node);
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
        return err;
    }
    
    // Update entry point if this node is at a higher level
    if (node->level > index->max_level) {
        index->entry_point = node_id;
        index->max_level = node->level;
    }
    
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

static cvector_error_t hnsw_connect_layers(hnsw_index_t* index, uint32_t node_id, uint32_t node_level) {
    // Legacy function - redirect to safe implementation
    return hnsw_connect_layers_safe(index, node_id, node_level);
}

cvector_error_t hnsw_search(hnsw_index_t* index, const float* query_vector, 
                           uint32_t top_k, hnsw_search_result_t** result) {
    return hnsw_search_with_ef(index, query_vector, top_k, index->ef_search, result);
}

cvector_error_t hnsw_search_with_ef(hnsw_index_t* index, const float* query_vector,
                                   uint32_t top_k, uint32_t ef, hnsw_search_result_t** result) {
    // Input validation with detailed logging
    cvector_error_t err = hnsw_validate_index(index);
    if (err != CVECTOR_SUCCESS) return err;
    
    if (!query_vector) {
        hnsw_log(HNSW_LOG_ERROR, "Query vector is NULL");
        return CVECTOR_ERROR_INVALID_ARGS;
    }
    
    if (!result) {
        hnsw_log(HNSW_LOG_ERROR, "Result pointer is NULL");
        return CVECTOR_ERROR_INVALID_ARGS;
    }
    
    if (top_k == 0) {
        hnsw_log(HNSW_LOG_ERROR, "top_k must be greater than 0");
        return CVECTOR_ERROR_INVALID_ARGS;
    }
    
    if (ef == 0) {
        hnsw_log(HNSW_LOG_WARN, "ef is 0, using default value %d", index->ef_search);
        ef = index->ef_search;
    }
    
    hnsw_log(HNSW_LOG_DEBUG, "Searching HNSW index: top_k=%u, ef=%u, nodes=%u", 
             top_k, ef, index->node_count);
    
    if (index->node_count == 0 || index->entry_point == UINT32_MAX) {
        hnsw_log(HNSW_LOG_INFO, "Empty index, returning empty results");
        *result = calloc(1, sizeof(hnsw_search_result_t));
        if (!*result) {
            hnsw_log(HNSW_LOG_ERROR, "Failed to allocate empty result structure");
            return CVECTOR_ERROR_OUT_OF_MEMORY;
        }
        return CVECTOR_SUCCESS;
    }
    
    index->search_count++;
    
    hnsw_priority_queue_t* entry_points;
    err = hnsw_pq_create(ef, false, &entry_points);
    if (err != CVECTOR_SUCCESS) return err;
    
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
            return err;
        }
    }
    
    // Search at level 0 with ef
    err = hnsw_search_layer(index, query_vector, entry_points, ef, 0);
    if (err != CVECTOR_SUCCESS) {
        hnsw_pq_destroy(entry_points);
        return err;
    }
    
    // Create result structure
    *result = malloc(sizeof(hnsw_search_result_t));
    if (!*result) {
        hnsw_pq_destroy(entry_points);
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
            return CVECTOR_ERROR_OUT_OF_MEMORY;
        }
        
        // Extract results from priority queue
        for (uint32_t i = 0; i < result_count; i++) {
            uint32_t node_id;
            float distance;
            if (hnsw_pq_pop(entry_points, &node_id, &distance)) {
                temp_results[i].id = index->nodes[node_id]->id;
                temp_results[i].similarity = distance;
                index->total_distance_computations++;
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
        hnsw_log(HNSW_LOG_ERROR, "Filepath is NULL");
        return CVECTOR_ERROR_INVALID_ARGS;
    }
    
    hnsw_log(HNSW_LOG_INFO, "Saving HNSW index to %s", filepath);
    
    FILE* file = fopen(filepath, "wb");
    if (!file) {
        hnsw_log(HNSW_LOG_ERROR, "Failed to open file %s for writing", filepath);
        return CVECTOR_ERROR_FILE_IO;
    }
    
    // Write header with magic number and version
    uint32_t magic = 0x484E5357; // "HNSW"
    uint32_t version = 1;
    if (fwrite(&magic, sizeof(magic), 1, file) != 1 ||
        fwrite(&version, sizeof(version), 1, file) != 1) {
        hnsw_log(HNSW_LOG_ERROR, "Failed to write header");
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
        hnsw_log(HNSW_LOG_ERROR, "Failed to write index metadata");
        fclose(file);
        return CVECTOR_ERROR_FILE_IO;
    }
    
    // Write nodes
    for (uint32_t i = 0; i < index->node_count; i++) {
        if (!index->nodes[i]) {
            hnsw_log(HNSW_LOG_ERROR, "Node %u is NULL during save", i);
            fclose(file);
            return CVECTOR_ERROR_DB_CORRUPT;
        }
        
        hnsw_node_t* node = index->nodes[i];
        
        // Write node metadata
        if (fwrite(&node->id, sizeof(node->id), 1, file) != 1 ||
            fwrite(&node->level, sizeof(node->level), 1, file) != 1 ||
            fwrite(&node->dimension, sizeof(node->dimension), 1, file) != 1) {
            hnsw_log(HNSW_LOG_ERROR, "Failed to write node %u metadata", i);
            fclose(file);
            return CVECTOR_ERROR_FILE_IO;
        }
        
        // Write vector data
        if (fwrite(node->vector_data, sizeof(float), node->dimension, file) != node->dimension) {
            hnsw_log(HNSW_LOG_ERROR, "Failed to write node %u vector data", i);
            fclose(file);
            return CVECTOR_ERROR_FILE_IO;
        }
        
        // Write connections for each level
        for (uint32_t level = 0; level <= node->level; level++) {
            if (fwrite(&node->connection_count[level], sizeof(node->connection_count[level]), 1, file) != 1) {
                hnsw_log(HNSW_LOG_ERROR, "Failed to write node %u level %u connection count", i, level);
                fclose(file);
                return CVECTOR_ERROR_FILE_IO;
            }
            
            if (node->connection_count[level] > 0) {
                if (fwrite(node->connections[level], sizeof(uint32_t), 
                          node->connection_count[level], file) != node->connection_count[level]) {
                    hnsw_log(HNSW_LOG_ERROR, "Failed to write node %u level %u connections", i, level);
                    fclose(file);
                    return CVECTOR_ERROR_FILE_IO;
                }
            }
        }
    }
    
    fclose(file);
    hnsw_log(HNSW_LOG_INFO, "Successfully saved HNSW index with %u nodes", index->node_count);
    return CVECTOR_SUCCESS;
}

cvector_error_t hnsw_load_index(const char* filepath, hnsw_index_t** index) {
    if (!filepath || !index) {
        hnsw_log(HNSW_LOG_ERROR, "Invalid arguments to hnsw_load_index");
        return CVECTOR_ERROR_INVALID_ARGS;
    }
    
    hnsw_log(HNSW_LOG_INFO, "Loading HNSW index from %s", filepath);
    
    FILE* file = fopen(filepath, "rb");
    if (!file) {
        hnsw_log(HNSW_LOG_ERROR, "Failed to open file %s for reading", filepath);
        return CVECTOR_ERROR_FILE_IO;
    }
    
    // Read and verify header
    uint32_t magic, version;
    if (fread(&magic, sizeof(magic), 1, file) != 1 ||
        fread(&version, sizeof(version), 1, file) != 1) {
        hnsw_log(HNSW_LOG_ERROR, "Failed to read header");
        fclose(file);
        return CVECTOR_ERROR_FILE_IO;
    }
    
    if (magic != 0x484E5357) {
        hnsw_log(HNSW_LOG_ERROR, "Invalid magic number: expected 0x484E5357, got 0x%08X", magic);
        fclose(file);
        return CVECTOR_ERROR_DB_CORRUPT;
    }
    
    if (version != 1) {
        hnsw_log(HNSW_LOG_ERROR, "Unsupported version: %u", version);
        fclose(file);
        return CVECTOR_ERROR_DB_CORRUPT;
    }
    
    // Read index metadata
    uint32_t dimension;
    cvector_similarity_t similarity_type;
    if (fread(&dimension, sizeof(dimension), 1, file) != 1 ||
        fread(&similarity_type, sizeof(similarity_type), 1, file) != 1) {
        hnsw_log(HNSW_LOG_ERROR, "Failed to read index basic metadata");
        fclose(file);
        return CVECTOR_ERROR_FILE_IO;
    }
    
    // Create index
    cvector_error_t err = hnsw_create_index(dimension, similarity_type, index);
    if (err != CVECTOR_SUCCESS) {
        hnsw_log(HNSW_LOG_ERROR, "Failed to create index during load");
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
        hnsw_log(HNSW_LOG_ERROR, "Failed to read index configuration");
        hnsw_destroy_index(idx);
        *index = NULL;
        fclose(file);
        return CVECTOR_ERROR_FILE_IO;
    }
    
    // Ensure capacity is sufficient
    while (idx->node_capacity < idx->node_count) {
        err = hnsw_resize_index(idx);
        if (err != CVECTOR_SUCCESS) {
            hnsw_log(HNSW_LOG_ERROR, "Failed to resize index during load");
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
            hnsw_log(HNSW_LOG_ERROR, "Failed to allocate node %u", i);
            hnsw_destroy_index(idx);
            *index = NULL;
            fclose(file);
            return CVECTOR_ERROR_OUT_OF_MEMORY;
        }
        
        // Read node metadata
        if (fread(&node->id, sizeof(node->id), 1, file) != 1 ||
            fread(&node->level, sizeof(node->level), 1, file) != 1 ||
            fread(&node->dimension, sizeof(node->dimension), 1, file) != 1) {
            hnsw_log(HNSW_LOG_ERROR, "Failed to read node %u metadata", i);
            free(node);
            hnsw_destroy_index(idx);
            *index = NULL;
            fclose(file);
            return CVECTOR_ERROR_FILE_IO;
        }
        
        // Allocate and read vector data
        node->vector_data = malloc(node->dimension * sizeof(float));
        if (!node->vector_data) {
            hnsw_log(HNSW_LOG_ERROR, "Failed to allocate vector data for node %u", i);
            free(node);
            hnsw_destroy_index(idx);
            *index = NULL;
            fclose(file);
            return CVECTOR_ERROR_OUT_OF_MEMORY;
        }
        
        if (fread(node->vector_data, sizeof(float), node->dimension, file) != node->dimension) {
            hnsw_log(HNSW_LOG_ERROR, "Failed to read node %u vector data", i);
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
                hnsw_log(HNSW_LOG_ERROR, "Failed to read node %u level %u connection count", i, level);
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
                    hnsw_log(HNSW_LOG_ERROR, "Failed to allocate connections for node %u level %u", i, level);
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
                    hnsw_log(HNSW_LOG_ERROR, "Failed to read node %u level %u connections", i, level);
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
    hnsw_log(HNSW_LOG_INFO, "Successfully loaded HNSW index with %u nodes", idx->node_count);
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
    
    return CVECTOR_SUCCESS;
}