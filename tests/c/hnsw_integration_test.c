#include "../../src/core/cvector.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
    printf("🔧 HNSW INTEGRATION TEST\n");
    
    // Create database
    const char* db_path = "./hnsw_integration.cvdb";
    remove(db_path);
    
    cvector_db_config_t config = {0};
    strcpy(config.name, "hnsw_test");
    strcpy(config.data_path, db_path);
    config.dimension = 4;
    config.default_similarity = CVECTOR_SIMILARITY_COSINE;
    config.memory_mapped = false;
    config.max_vectors = 100;
    
    cvector_db_t* db = NULL;
    cvector_error_t err = cvector_db_create(&config, &db);
    if (err != CVECTOR_SUCCESS) {
        printf("❌ Database creation failed: %s\n", cvector_error_string(err));
        return 1;
    }
    printf("✅ Database created\n");
    
    // Insert vectors one by one and observe behavior
    float vectors[][4] = {
        {1.0f, 0.0f, 0.0f, 0.0f},   // ID 1
        {0.0f, 1.0f, 0.0f, 0.0f},   // ID 2  
        {0.0f, 0.0f, 1.0f, 0.0f},   // ID 3
        {0.5f, 0.5f, 0.0f, 0.0f},   // ID 4
        {0.9f, 0.1f, 0.0f, 0.0f}    // ID 5
    };
    
    printf("Inserting vectors with HNSW enabled...\n");
    for (int i = 0; i < 5; i++) {
        cvector_t* vector = NULL;
        err = cvector_create_vector(i + 1, 4, vectors[i], &vector);
        if (err != CVECTOR_SUCCESS) {
            printf("❌ Vector %d creation failed: %s\n", i+1, cvector_error_string(err));
            cvector_db_close(db);
            return 1;
        }
        
        printf("  Inserting vector %d: [%.1f, %.1f, %.1f, %.1f]\n", 
               i+1, vectors[i][0], vectors[i][1], vectors[i][2], vectors[i][3]);
        
        err = cvector_insert(db, vector);
        if (err != CVECTOR_SUCCESS) {
            printf("❌ Vector %d insertion failed: %s\n", i+1, cvector_error_string(err));
            cvector_free_vector(vector);
            cvector_db_close(db);
            return 1;
        }
        printf("  ✅ Vector %d inserted successfully\n", i+1);
        
        cvector_free_vector(vector);
    }
    
    // Test search
    printf("\nTesting HNSW search...\n");
    cvector_query_t query = {0};
    query.query_vector = vectors[0]; // Search for [1,0,0,0]
    query.dimension = 4;
    query.top_k = 3;
    query.similarity = CVECTOR_SIMILARITY_COSINE;
    query.min_similarity = 0.0f;
    
    cvector_result_t* results = NULL;
    size_t result_count = 0;
    
    err = cvector_search(db, &query, &results, &result_count);
    if (err != CVECTOR_SUCCESS) {
        printf("❌ Search failed: %s\n", cvector_error_string(err));
        cvector_db_close(db);
        return 1;
    }
    
    printf("✅ Search succeeded, found %zu results:\n", result_count);
    for (size_t i = 0; i < result_count; i++) {
        printf("  Rank %zu: ID=%llu, Similarity=%.6f\n", 
               i+1, results[i].id, results[i].similarity);
    }
    
    if (results) {
        cvector_free_results(results, result_count);
    }
    
    // Test database persistence and HNSW rebuild
    printf("\nTesting database persistence and HNSW rebuild...\n");
    err = cvector_db_close(db);
    if (err != CVECTOR_SUCCESS) {
        printf("❌ Database close failed: %s\n", cvector_error_string(err));
        return 1;
    }
    printf("✅ Database closed\n");
    
    // Reopen database
    cvector_db_t* reopened_db = NULL;
    err = cvector_db_open(db_path, &reopened_db);
    if (err != CVECTOR_SUCCESS) {
        printf("❌ Database reopen failed: %s\n", cvector_error_string(err));
        return 1;
    }
    printf("✅ Database reopened (HNSW should be rebuilt)\n");
    
    // Test search after reopen
    err = cvector_search(reopened_db, &query, &results, &result_count);
    if (err != CVECTOR_SUCCESS) {
        printf("❌ Search after reopen failed: %s\n", cvector_error_string(err));
        cvector_db_close(reopened_db);
        return 1;
    }
    
    printf("✅ Search after reopen succeeded, found %zu results:\n", result_count);
    for (size_t i = 0; i < result_count; i++) {
        printf("  Rank %zu: ID=%llu, Similarity=%.6f\n", 
               i+1, results[i].id, results[i].similarity);
    }
    
    if (results) {
        cvector_free_results(results, result_count);
    }
    
    cvector_db_close(reopened_db);
    remove(db_path);
    
    printf("\n🎉 HNSW integration test completed successfully!\n");
    return 0;
}