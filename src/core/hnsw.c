
#include "hnsw.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Random number generator for level selection
static uint32_t hnsw_random_seed = 1;

static uint32_t hnsw_random() {
    hnsw_random_seed = hnsw_random_seed * 1103515245 + 12345;
    return hnsw_random_seed;
}

static void hnsw_init_random() {
    hnsw_random_seed = (uint32_t)time(NULL);
}

// Level generation using exponential decay
static uint32_t hnsw_get_random_level(float ml) {
    uint32_t level = 0;
    while (((float)hnsw_random() / UINT32_MAX) < (1.0f / exp(level / ml)) && level < HNSW_MAX_LEVEL - 1) {
        level++;
    }
    return level;
}

// Similarity calculation
float hnsw_calculate_similarity(const float* a, const float* b, uint32_t dimension, 
                               cvector_similarity_t similarity_type) {
    switch (similarity_type) {
        case CVECTOR_SIMILARITY_COSINE: {
            float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
            for (uint32_t i = 0; i < dimension; i++) {
                dot += a[i] * b[i];
                norm_a += a[i] * a[i];
                norm_b += b[i] * b[i];
            }
            float norm_product = sqrtf(norm_a * norm_b);
            return (norm_product > 0.0f) ? (dot / norm_product) : 0.0f;
        }
        
        case CVECTOR_SIMILARITY_DOT_PRODUCT: {
            float dot = 0.0f;
            for (uint32_t i = 0; i < dimension; i++) {
                dot += a[i] * b[i];
            }
            return dot;
        }
        
        case CVECTOR_SIMILARITY_EUCLIDEAN: {
            float sum = 0.0f;
            for (uint32_t i = 0; i < dimension; i++) {
                float diff = a[i] - b[i];
                sum += diff * diff;
            }
            return -sqrtf(sum); // Negative distance for similarity
        }
        
        default:
            return 0.0f;
    }
}

// Priority Queue Implementation
cvector_error_t hnsw_pq_create(uint32_t capacity, bool is_max_heap, hnsw_priority_queue_t** pq) {
    *pq = malloc(sizeof(hnsw_priority_queue_t));
    if (!*pq) return CVECTOR_ERROR_OUT_OF_MEMORY;
    
    (*pq)->items = malloc(capacity * sizeof((*pq)->items[0]));
    if (!(*pq)->items) {
        free(*pq);
        return CVECTOR_ERROR_OUT_OF_MEMORY;
    }
    
    (*pq)->count = 0;
    (*pq)->capacity = capacity;
    (*pq)->is_max_heap = is_max_heap;
    
    return CVECTOR_SUCCESS;
}

void hnsw_pq_destroy(hnsw_priority_queue_t* pq) {
    if (pq) {
        free(pq->items);
        free(pq);
    }
}

static void hnsw_pq_heapify_up(hnsw_priority_queue_t* pq, uint32_t index) {
    if (index == 0) return;
    
    uint32_t parent = (index - 1) / 2;
    bool should_swap = pq->is_max_heap ? 
        (pq->items[index].distance > pq->items[parent].distance) :
        (pq->items[index].distance < pq->items[parent].distance);
    
    if (should_swap) {
        // Swap
        struct { uint32_t node_id; float distance; } temp = pq->items[index];
        pq->items[index] = pq->items[parent];
        pq->items[parent] = temp;
        hnsw_pq_heapify_up(pq, parent);
    }
}

static void hnsw_pq_heapify_down(hnsw_priority_queue_t* pq, uint32_t index) {
    uint32_t left = 2 * index + 1;
    uint32_t right = 2 * index + 2;
    uint32_t target = index;
    
    if (left < pq->count) {
        bool left_better = pq->is_max_heap ?
            (pq->items[left].distance > pq->items[target].distance) :
            (pq->items[left].distance < pq->items[target].distance);
        if (left_better) target = left;
    }
    
    if (right < pq->count) {
        bool right_better = pq->is_max_heap ?
            (pq->items[right].distance > pq->items[target].distance) :
            (pq->items[right].distance < pq->items[target].distance);
        if (right_better) target = right;
    }
    
    if (target != index) {
        // Swap
        struct { uint32_t node_id; float distance; } temp = pq->items[index];
        pq->items[index] = pq->items[target];
        pq->items[target] = temp;
        hnsw_pq_heapify_down(pq, target);
    }
}

cvector_error_t hnsw_pq_push(hnsw_priority_queue_t* pq, uint32_t node_id, float distance) {
    if (pq->count >= pq->capacity) {
        return CVECTOR_ERROR_OUT_OF_MEMORY; // Queue full
    }
    
    pq->items[pq->count].node_id = node_id;
    pq->items[pq->count].distance = distance;
    hnsw_pq_heapify_up(pq, pq->count);
    pq->count++;
    
    return CVECTOR_SUCCESS;
}

bool hnsw_pq_pop(hnsw_priority_queue_t* pq, uint32_t* node_id, float* distance) {
    if (pq->count == 0) return false;
    
    *node_id = pq->items[0].node_id;
    *distance = pq->items[0].distance;
    
    pq->count--;
    if (pq->count > 0) {
        pq->items[0] = pq->items[pq->count];
        hnsw_pq_heapify_down(pq, 0);
    }
    
    return true;
}

bool hnsw_pq_is_empty(hnsw_priority_queue_t* pq) {
    return pq->count == 0;
}

bool hnsw_pq_is_full(hnsw_priority_queue_t* pq) {
    return pq->count >= pq->capacity;
}

// HNSW Index Implementation
cvector_error_t hnsw_create_index(uint32_t dimension, cvector_similarity_t similarity_type, 
                                  hnsw_index_t** index) {
    *index = malloc(sizeof(hnsw_index_t));
    if (!*index) return CVECTOR_ERROR_OUT_OF_MEMORY;
    
    memset(*index, 0, sizeof(hnsw_index_t));
    
    (*index)->dimension = dimension;
    (*index)->similarity_type = similarity_type;
    (*index)->M = HNSW_DEFAULT_M;
    (*index)->ef_construction = HNSW_DEFAULT_EF_CONSTRUCTION;
    (*index)->ef_search = HNSW_DEFAULT_EF_SEARCH;
    (*index)->ml = HNSW_DEFAULT_ML;
    (*index)->entry_point = UINT32_MAX; // No entry point initially
    (*index)->max_level = 0;
    
    // Initialize with small capacity
    (*index)->node_capacity = 1000;
    (*index)->nodes = malloc((*index)->node_capacity * sizeof(hnsw_node_t*));
    if (!(*index)->nodes) {
        free(*index);
        return CVECTOR_ERROR_OUT_OF_MEMORY;
    }
    
    hnsw_init_random();
    
    return CVECTOR_SUCCESS;
}

cvector_error_t hnsw_destroy_index(hnsw_index_t* index) {
    if (!index) return CVECTOR_SUCCESS;
    
    // Free all nodes
    for (uint32_t i = 0; i < index->node_count; i++) {
        if (index->nodes[i]) {
            // Free connections for each level
            for (uint32_t level = 0; level <= index->nodes[i]->level; level++) {
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

static cvector_error_t hnsw_resize_nodes(hnsw_index_t* index) {
    if (index->node_count >= index->node_capacity) {
        uint32_t new_capacity = index->node_capacity * 2;
        hnsw_node_t** new_nodes = realloc(index->nodes, new_capacity * sizeof(hnsw_node_t*));
        if (!new_nodes) return CVECTOR_ERROR_OUT_OF_MEMORY;
        
        index->nodes = new_nodes;
        index->node_capacity = new_capacity;
    }
    return CVECTOR_SUCCESS;
}

static hnsw_node_t* hnsw_create_node(cvector_id_t id, const float* vector, uint32_t dimension, uint32_t level) {
    hnsw_node_t* node = malloc(sizeof(hnsw_node_t));
    if (!node) return NULL;
    
    memset(node, 0, sizeof(hnsw_node_t));
    node->id = id;
    node->level = level;
    node->dimension = dimension;
    
    // Allocate vector data
    node->vector_data = malloc(dimension * sizeof(float));
    if (!node->vector_data) {
        free(node);
        return NULL;
    }
    memcpy(node->vector_data, vector, dimension * sizeof(float));
    
    // Initialize connection arrays for each level
    for (uint32_t l = 0; l <= level; l++) {
        uint32_t max_connections = (l == 0) ? HNSW_DEFAULT_M * 2 : HNSW_DEFAULT_M;
        node->connections[l] = malloc(max_connections * sizeof(uint32_t));
        if (!node->connections[l]) {
            // Cleanup on failure
            for (uint32_t cleanup_l = 0; cleanup_l < l; cleanup_l++) {
                free(node->connections[cleanup_l]);
            }
            free(node->vector_data);
            free(node);
            return NULL;
        }
        node->connection_count[l] = 0;
    }
    
    return node;
}

// Find closest nodes at a specific level
static cvector_error_t hnsw_search_level(hnsw_index_t* index, const float* query_vector,
                                        uint32_t entry_point, uint32_t ef, uint32_t level,
                                        hnsw_priority_queue_t** candidates) {
    // Create priority queues
    hnsw_priority_queue_t* visited;
    hnsw_priority_queue_t* candidates_pq;
    hnsw_priority_queue_t* dynamic_candidates;
    
    cvector_error_t err;
    err = hnsw_pq_create(ef * 2, false, &visited);
    if (err != CVECTOR_SUCCESS) return err;
    
    err = hnsw_pq_create(ef, true, &candidates_pq);
    if (err != CVECTOR_SUCCESS) {
        hnsw_pq_destroy(visited);
        return err;
    }
    
    err = hnsw_pq_create(ef * 2, false, &dynamic_candidates);
    if (err != CVECTOR_SUCCESS) {
        hnsw_pq_destroy(visited);
        hnsw_pq_destroy(candidates_pq);
        return err;
    }
    
    // Calculate distance to entry point
    float entry_distance = hnsw_calculate_similarity(query_vector, 
        index->nodes[entry_point]->vector_data, index->dimension, index->similarity_type);
    
    // Initialize with entry point
    hnsw_pq_push(visited, entry_point, entry_distance);
    hnsw_pq_push(candidates_pq, entry_point, entry_distance);
    hnsw_pq_push(dynamic_candidates, entry_point, entry_distance);
    
    // Search
    while (!hnsw_pq_is_empty(dynamic_candidates)) {
        uint32_t current_node;
        float current_distance;
        
        if (!hnsw_pq_pop(dynamic_candidates, &current_node, &current_distance)) {
            break;
        }
        
        // Check if we should continue (early termination)
        if (!hnsw_pq_is_empty(candidates_pq)) {
            float worst_candidate_distance = candidates_pq->items[0].distance;
            if (current_distance > worst_candidate_distance) {
                break;
            }
        }
        
        // Explore neighbors
        hnsw_node_t* node = index->nodes[current_node];
        for (uint32_t i = 0; i < node->connection_count[level]; i++) {
            uint32_t neighbor = node->connections[level][i];
            
            // Check if already visited
            bool already_visited = false;
            for (uint32_t v = 0; v < visited->count; v++) {
                if (visited->items[v].node_id == neighbor) {
                    already_visited = true;
                    break;
                }
            }
            
            if (!already_visited) {
                float neighbor_distance = hnsw_calculate_similarity(query_vector,
                    index->nodes[neighbor]->vector_data, index->dimension, index->similarity_type);
                
                index->total_distance_computations++;
                
                hnsw_pq_push(visited, neighbor, neighbor_distance);
                
                // Add to candidates if better than worst or queue not full
                if (candidates_pq->count < ef || neighbor_distance > candidates_pq->items[0].distance) {
                    hnsw_pq_push(candidates_pq, neighbor, neighbor_distance);
                    hnsw_pq_push(dynamic_candidates, neighbor, neighbor_distance);
                    
                    // Remove worst if queue is full
                    if (candidates_pq->count > ef) {
                        uint32_t worst_id;
                        float worst_distance;
                        hnsw_pq_pop(candidates_pq, &worst_id, &worst_distance);
                    }
                }
            }
        }
    }
    
    *candidates = candidates_pq;
    
    hnsw_pq_destroy(visited);
    hnsw_pq_destroy(dynamic_candidates);
    
    return CVECTOR_SUCCESS;
}

cvector_error_t hnsw_add_vector(hnsw_index_t* index, cvector_id_t id, const float* vector) {
    if (!index || !vector) return CVECTOR_ERROR_INVALID_ARGS;
    
    cvector_error_t err = hnsw_resize_nodes(index);
    if (err != CVECTOR_SUCCESS) return err;
    
    // Generate random level for new node
    uint32_t level = hnsw_get_random_level(index->ml);
    
    // Create new node
    hnsw_node_t* new_node = hnsw_create_node(id, vector, index->dimension, level);
    if (!new_node) return CVECTOR_ERROR_OUT_OF_MEMORY;
    
    uint32_t node_index = index->node_count;
    index->nodes[node_index] = new_node;
    index->node_count++;
    
    // If this is the first node, make it the entry point
    if (index->entry_point == UINT32_MAX) {
        index->entry_point = node_index;
        index->max_level = level;
        return CVECTOR_SUCCESS;
    }
    
    // Update max level if necessary
    if (level > index->max_level) {
        index->max_level = level;
        index->entry_point = node_index;
    }
    
    // Search for closest nodes and create connections
    uint32_t entry_point = index->entry_point;
    
    // Search from top level down to level+1
    for (int32_t lc = (int32_t)index->max_level; lc > (int32_t)level; lc--) {
        hnsw_priority_queue_t* candidates;
        err = hnsw_search_level(index, vector, entry_point, 1, (uint32_t)lc, &candidates);
        if (err != CVECTOR_SUCCESS) {
            return err;
        }
        
        if (!hnsw_pq_is_empty(candidates)) {
            uint32_t closest;
            float distance;
            hnsw_pq_pop(candidates, &closest, &distance);
            entry_point = closest;
        }
        
        hnsw_pq_destroy(candidates);
    }
    
    // Search and connect at each level from level down to 0
    for (int32_t lc = (int32_t)level; lc >= 0; lc--) {
        hnsw_priority_queue_t* candidates;
        uint32_t ef = (lc == 0) ? index->ef_construction : index->ef_construction;
        
        err = hnsw_search_level(index, vector, entry_point, ef, (uint32_t)lc, &candidates);
        if (err != CVECTOR_SUCCESS) {
            return err;
        }
        
        // Select neighbors and create bidirectional connections
        uint32_t max_connections = (lc == 0) ? index->M * 2 : index->M;
        uint32_t connection_count = 0;
        
        while (!hnsw_pq_is_empty(candidates) && connection_count < max_connections) {
            uint32_t neighbor;
            float distance;
            if (hnsw_pq_pop(candidates, &neighbor, &distance)) {
                // Add connection from new node to neighbor
                new_node->connections[lc][connection_count] = neighbor;
                connection_count++;
                
                // Add connection from neighbor to new node (bidirectional)
                hnsw_node_t* neighbor_node = index->nodes[neighbor];
                if (neighbor_node->connection_count[lc] < max_connections) {
                    neighbor_node->connections[lc][neighbor_node->connection_count[lc]] = node_index;
                    neighbor_node->connection_count[lc]++;
                }
            }
        }
        
        new_node->connection_count[lc] = connection_count;
        hnsw_pq_destroy(candidates);
    }
    
    return CVECTOR_SUCCESS;
}

cvector_error_t hnsw_search(hnsw_index_t* index, const float* query_vector, 
                           uint32_t top_k, hnsw_search_result_t** result) {
    return hnsw_search_with_ef(index, query_vector, top_k, index->ef_search, result);
}

cvector_error_t hnsw_search_with_ef(hnsw_index_t* index, const float* query_vector,
                                   uint32_t top_k, uint32_t ef, hnsw_search_result_t** result) {
    if (!index || !query_vector || !result || index->entry_point == UINT32_MAX) {
        return CVECTOR_ERROR_INVALID_ARGS;
    }
    
    index->search_count++;
    
    // Search from top level down to level 1
    uint32_t entry_point = index->entry_point;
    for (int32_t level = (int32_t)index->max_level; level > 0; level--) {
        hnsw_priority_queue_t* candidates;
        cvector_error_t err = hnsw_search_level(index, query_vector, entry_point, 1, 
                                               (uint32_t)level, &candidates);
        if (err != CVECTOR_SUCCESS) return err;
        
        if (!hnsw_pq_is_empty(candidates)) {
            uint32_t closest;
            float distance;
            hnsw_pq_pop(candidates, &closest, &distance);
            entry_point = closest;
        }
        
        hnsw_pq_destroy(candidates);
    }
    
    // Search at level 0 with higher ef
    hnsw_priority_queue_t* candidates;
    cvector_error_t err = hnsw_search_level(index, query_vector, entry_point, 
                                           ef > top_k ? ef : top_k, 0, &candidates);
    if (err != CVECTOR_SUCCESS) return err;
    
    // Create result structure
    *result = malloc(sizeof(hnsw_search_result_t));
    if (!*result) {
        hnsw_pq_destroy(candidates);
        return CVECTOR_ERROR_OUT_OF_MEMORY;
    }
    
    uint32_t result_count = candidates->count < top_k ? candidates->count : top_k;
    (*result)->count = result_count;
    (*result)->capacity = result_count;
    
    (*result)->ids = malloc(result_count * sizeof(cvector_id_t));
    (*result)->similarities = malloc(result_count * sizeof(float));
    
    if (!(*result)->ids || !(*result)->similarities) {
        free((*result)->ids);
        free((*result)->similarities);
        free(*result);
        hnsw_pq_destroy(candidates);
        return CVECTOR_ERROR_OUT_OF_MEMORY;
    }
    
    // Extract top-k results (candidates is a max-heap, so we get best results first)
    for (uint32_t i = 0; i < result_count; i++) {
        uint32_t node_id;
        float similarity;
        if (hnsw_pq_pop(candidates, &node_id, &similarity)) {
            (*result)->ids[result_count - 1 - i] = index->nodes[node_id]->id; // Reverse order
            (*result)->similarities[result_count - 1 - i] = similarity;
        }
    }
    
    hnsw_pq_destroy(candidates);
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
    if (!index || !stats) return CVECTOR_ERROR_INVALID_ARGS;
    
    stats->node_count = index->node_count;
    stats->max_level = index->max_level;
    stats->search_count = index->search_count;
    stats->distance_computations = index->total_distance_computations;
    stats->entry_point_level = (index->entry_point != UINT32_MAX) ? 
                               index->nodes[index->entry_point]->level : 0;
    
    // Calculate average connections
    if (index->node_count > 0) {
        uint64_t total_connections = 0;
        for (uint32_t i = 0; i < index->node_count; i++) {
            for (uint32_t level = 0; level <= index->nodes[i]->level; level++) {
                total_connections += index->nodes[i]->connection_count[level];
            }
        }
        stats->avg_connections_per_node = (float)total_connections / index->node_count;
    } else {
        stats->avg_connections_per_node = 0.0f;
    }
    
    return CVECTOR_SUCCESS;
}

cvector_error_t hnsw_set_config(hnsw_index_t* index, const hnsw_config_t* config) {
    if (!index || !config) return CVECTOR_ERROR_INVALID_ARGS;
    
    index->M = config->M;
    index->ef_construction = config->ef_construction;
    index->ef_search = config->ef_search;
    index->ml = config->ml;
    
    return CVECTOR_SUCCESS;
}

cvector_error_t hnsw_get_config(hnsw_index_t* index, hnsw_config_t* config) {
    if (!index || !config) return CVECTOR_ERROR_INVALID_ARGS;
    
    config->M = index->M;
    config->ef_construction = index->ef_construction;
    config->ef_search = index->ef_search;
    config->ml = index->ml;
    
    return CVECTOR_SUCCESS;
}

cvector_error_t hnsw_remove_vector(hnsw_index_t* index, cvector_id_t id) {
    if (!index) return CVECTOR_ERROR_INVALID_ARGS;
    
    // Find the node to remove
    uint32_t node_to_remove = UINT32_MAX;
    for (uint32_t i = 0; i < index->node_count; i++) {
        if (index->nodes[i]->id == id) {
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
        if (i == node_to_remove) continue;
        
        hnsw_node_t* other_node = index->nodes[i];
        for (uint32_t level = 0; level <= other_node->level; level++) {
            // Remove connection to the node being deleted
            for (uint32_t j = 0; j < other_node->connection_count[level]; j++) {
                if (other_node->connections[level][j] == node_to_remove) {
                    // Shift remaining connections
                    for (uint32_t k = j; k < other_node->connection_count[level] - 1; k++) {
                        other_node->connections[level][k] = other_node->connections[level][k + 1];
                    }
                    other_node->connection_count[level]--;
                    break;
                }
            }
            
            // Update connection indices (nodes after removed node shift down)
            for (uint32_t j = 0; j < other_node->connection_count[level]; j++) {
                if (other_node->connections[level][j] > node_to_remove) {
                    other_node->connections[level][j]--;
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
    
    // Shift remaining nodes
    for (uint32_t i = node_to_remove; i < index->node_count - 1; i++) {
        index->nodes[i] = index->nodes[i + 1];
    }
    index->node_count--;
    
    // Update entry point if necessary
    if (node_to_remove == index->entry_point) {
        if (index->node_count > 0) {
            // Find new entry point (node with highest level)
            uint32_t new_entry = 0;
            uint32_t max_level = index->nodes[0]->level;
            for (uint32_t i = 1; i < index->node_count; i++) {
                if (index->nodes[i]->level > max_level) {
                    max_level = index->nodes[i]->level;
                    new_entry = i;
                }
            }
            index->entry_point = new_entry;
            index->max_level = max_level;
        } else {
            index->entry_point = UINT32_MAX;
            index->max_level = 0;
        }
    } else if (node_to_remove < index->entry_point) {
        index->entry_point--;
    }
    
    return CVECTOR_SUCCESS;
}

// Persistence functions (basic implementation)
cvector_error_t hnsw_save_index(hnsw_index_t* index, const char* filepath) {
    if (!index || !filepath) return CVECTOR_ERROR_INVALID_ARGS;
    
    FILE* file = fopen(filepath, "wb");
    if (!file) return CVECTOR_ERROR_FILE_IO;
    
    // Write header
    uint32_t magic = 0x484E5357; // "HNSW"
    uint32_t version = 1;
    fwrite(&magic, sizeof(magic), 1, file);
    fwrite(&version, sizeof(version), 1, file);
    fwrite(&index->dimension, sizeof(index->dimension), 1, file);
    fwrite(&index->similarity_type, sizeof(index->similarity_type), 1, file);
    fwrite(&index->node_count, sizeof(index->node_count), 1, file);
    fwrite(&index->entry_point, sizeof(index->entry_point), 1, file);
    fwrite(&index->max_level, sizeof(index->max_level), 1, file);
    fwrite(&index->M, sizeof(index->M), 1, file);
    fwrite(&index->ef_construction, sizeof(index->ef_construction), 1, file);
    fwrite(&index->ef_search, sizeof(index->ef_search), 1, file);
    fwrite(&index->ml, sizeof(index->ml), 1, file);
    
    // Write nodes
    for (uint32_t i = 0; i < index->node_count; i++) {
        hnsw_node_t* node = index->nodes[i];
        
        fwrite(&node->id, sizeof(node->id), 1, file);
        fwrite(&node->level, sizeof(node->level), 1, file);
        fwrite(&node->dimension, sizeof(node->dimension), 1, file);
        fwrite(node->vector_data, sizeof(float), node->dimension, file);
        
        // Write connections for each level
        for (uint32_t level = 0; level <= node->level; level++) {
            fwrite(&node->connection_count[level], sizeof(node->connection_count[level]), 1, file);
            fwrite(node->connections[level], sizeof(uint32_t), node->connection_count[level], file);
        }
    }
    
    fclose(file);
    return CVECTOR_SUCCESS;
}

cvector_error_t hnsw_load_index(const char* filepath, hnsw_index_t** index) {
    if (!filepath || !index) return CVECTOR_ERROR_INVALID_ARGS;
    
    FILE* file = fopen(filepath, "rb");
    if (!file) return CVECTOR_ERROR_FILE_IO;
    
    // Read and verify header
    uint32_t magic, version;
    fread(&magic, sizeof(magic), 1, file);
    fread(&version, sizeof(version), 1, file);
    
    if (magic != 0x484E5357 || version != 1) {
        fclose(file);
        return CVECTOR_ERROR_DB_CORRUPT;
    }
    
    // Read index metadata
    uint32_t dimension;
    cvector_similarity_t similarity_type;
    fread(&dimension, sizeof(dimension), 1, file);
    fread(&similarity_type, sizeof(similarity_type), 1, file);
    
    // Create index
    cvector_error_t err = hnsw_create_index(dimension, similarity_type, index);
    if (err != CVECTOR_SUCCESS) {
        fclose(file);
        return err;
    }
    
    // Read configuration
    fread(&(*index)->node_count, sizeof((*index)->node_count), 1, file);
    fread(&(*index)->entry_point, sizeof((*index)->entry_point), 1, file);
    fread(&(*index)->max_level, sizeof((*index)->max_level), 1, file);
    fread(&(*index)->M, sizeof((*index)->M), 1, file);
    fread(&(*index)->ef_construction, sizeof((*index)->ef_construction), 1, file);
    fread(&(*index)->ef_search, sizeof((*index)->ef_search), 1, file);
    fread(&(*index)->ml, sizeof((*index)->ml), 1, file);
    
    // Resize node array if needed
    if ((*index)->node_count > (*index)->node_capacity) {
        (*index)->node_capacity = (*index)->node_count;
        (*index)->nodes = realloc((*index)->nodes, (*index)->node_capacity * sizeof(hnsw_node_t*));
        if (!(*index)->nodes) {
            hnsw_destroy_index(*index);
            fclose(file);
            return CVECTOR_ERROR_OUT_OF_MEMORY;
        }
    }
    
    // Read nodes
    for (uint32_t i = 0; i < (*index)->node_count; i++) {
        hnsw_node_t* node = malloc(sizeof(hnsw_node_t));
        if (!node) {
            hnsw_destroy_index(*index);
            fclose(file);
            return CVECTOR_ERROR_OUT_OF_MEMORY;
        }
        
        fread(&node->id, sizeof(node->id), 1, file);
        fread(&node->level, sizeof(node->level), 1, file);
        fread(&node->dimension, sizeof(node->dimension), 1, file);
        
        // Read vector data
        node->vector_data = malloc(node->dimension * sizeof(float));
        if (!node->vector_data) {
            free(node);
            hnsw_destroy_index(*index);
            fclose(file);
            return CVECTOR_ERROR_OUT_OF_MEMORY;
        }
        fread(node->vector_data, sizeof(float), node->dimension, file);
        
        // Read connections
        for (uint32_t level = 0; level <= node->level; level++) {
            fread(&node->connection_count[level], sizeof(node->connection_count[level]), 1, file);
            
            uint32_t max_connections = (level == 0) ? (*index)->M * 2 : (*index)->M;
            node->connections[level] = malloc(max_connections * sizeof(uint32_t));
            if (!node->connections[level]) {
                // Cleanup
                for (uint32_t cleanup_level = 0; cleanup_level < level; cleanup_level++) {
                    free(node->connections[cleanup_level]);
                }
                free(node->vector_data);
                free(node);
                hnsw_destroy_index(*index);
                fclose(file);
                return CVECTOR_ERROR_OUT_OF_MEMORY;
            }
            
            fread(node->connections[level], sizeof(uint32_t), node->connection_count[level], file);
        }
        
        (*index)->nodes[i] = node;
    }
    
    fclose(file);
    return CVECTOR_SUCCESS;
}