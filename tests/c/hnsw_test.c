#include "../../src/core/hnsw.h"
#include "../../src/core/similarity.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
    printf("🔧 HNSW ISOLATED TEST\n");
    
    // Test 1: Create HNSW index
    hnsw_index_t* index = NULL;
    cvector_error_t err = hnsw_create_index(4, CVECTOR_SIMILARITY_COSINE, &index);
    if (err != CVECTOR_SUCCESS) {
        printf("❌ HNSW index creation failed\n");
        return 1;
    }
    printf("✅ HNSW index created\n");
    
    // Test 2: Add first vector (should work)
    float vector1[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    printf("Adding first vector...\n");
    err = hnsw_add_vector(index, 1, vector1);
    if (err != CVECTOR_SUCCESS) {
        printf("❌ First vector addition failed\n");
        hnsw_destroy_index(index);
        return 1;
    }
    printf("✅ First vector added successfully\n");
    
    // Test 3: Add second vector (this usually causes segfault)
    float vector2[4] = {0.0f, 1.0f, 0.0f, 0.0f};
    printf("Adding second vector...\n");
    err = hnsw_add_vector(index, 2, vector2);
    if (err != CVECTOR_SUCCESS) {
        printf("❌ Second vector addition failed\n");
        hnsw_destroy_index(index);
        return 1;
    }
    printf("✅ Second vector added successfully\n");
    
    // Test 4: Add third vector
    float vector3[4] = {0.5f, 0.5f, 0.0f, 0.0f};
    printf("Adding third vector...\n");
    err = hnsw_add_vector(index, 3, vector3);
    if (err != CVECTOR_SUCCESS) {
        printf("❌ Third vector addition failed\n");
        hnsw_destroy_index(index);
        return 1;
    }
    printf("✅ Third vector added successfully\n");
    
    // Test 5: Search
    printf("Testing search...\n");
    hnsw_search_result_t* result = NULL;
    err = hnsw_search(index, vector1, 3, &result);
    if (err != CVECTOR_SUCCESS) {
        printf("❌ Search failed\n");
        hnsw_destroy_index(index);
        return 1;
    }
    
    printf("✅ Search successful, found %u results:\n", result->count);
    for (uint32_t i = 0; i < result->count; i++) {
        printf("   ID=%llu, Similarity=%.6f\n", result->ids[i], result->similarities[i]);
    }
    
    hnsw_free_search_result(result);
    hnsw_destroy_index(index);
    
    printf("🎉 HNSW test completed successfully!\n");
    return 0;
}