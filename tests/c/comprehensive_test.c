#include "../../src/core/cvector.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <sys/time.h>
#include <math.h>

// Helper function to get current time in milliseconds
double get_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

// Test result structure
typedef struct {
    int passed;
    int failed;
    double total_time_ms;
    char* error_msg;
} test_result_t;

// Global test stats
test_result_t global_stats = {0, 0, 0.0, NULL};

void test_assert(int condition, const char* test_name) {
    if (condition) {
        printf("✅ PASS: %s\n", test_name);
        global_stats.passed++;
    } else {
        printf("❌ FAIL: %s\n", test_name);
        global_stats.failed++;
    }
}

void print_separator(const char* title) {
    printf("\n==================== %s ====================\n", title);
}

int main() {
    printf("🚀 COMPREHENSIVE CVECTOR DATABASE TEST\n");
    printf("Testing all database operations and similarity search...\n");
    
    double start_time = get_time_ms();
    
    // Test database path
    const char* db_path = "./test_comprehensive.cvdb";
    
    // Remove existing database
    remove(db_path);
    
    print_separator("DATABASE CREATION");
    
    // Test 1: Create database
    cvector_db_config_t config = {0};
    strcpy(config.name, "comprehensive_test");
    strcpy(config.data_path, db_path);
    config.dimension = 4;
    config.default_similarity = CVECTOR_SIMILARITY_COSINE;
    config.memory_mapped = false;
    config.max_vectors = 10000;
    
    cvector_db_t* db = NULL;
    cvector_error_t err = cvector_db_create(&config, &db);
    test_assert(err == CVECTOR_SUCCESS && db != NULL, "Database creation");
    
    print_separator("VECTOR INSERTION");
    
    // Test 2-11: Insert test vectors
    float test_vectors[][4] = {
        {1.0f, 0.0f, 0.0f, 0.0f},   // ID 1: Unit vector X
        {0.0f, 1.0f, 0.0f, 0.0f},   // ID 2: Unit vector Y
        {0.0f, 0.0f, 1.0f, 0.0f},   // ID 3: Unit vector Z
        {0.0f, 0.0f, 0.0f, 1.0f},   // ID 4: Unit vector W
        {0.9f, 0.1f, 0.0f, 0.0f},   // ID 5: Similar to X
        {0.8f, 0.2f, 0.0f, 0.0f},   // ID 6: Similar to X
        {0.1f, 0.9f, 0.0f, 0.0f},   // ID 7: Similar to Y
        {0.5f, 0.5f, 0.0f, 0.0f},   // ID 8: Between X and Y
        {-1.0f, 0.0f, 0.0f, 0.0f},  // ID 9: Opposite of X
        {0.0f, 0.0f, 0.0f, 0.0f}    // ID 10: Zero vector
    };
    
    double insert_start = get_time_ms();
    
    for (int i = 0; i < 10; i++) {
        cvector_t* vector = NULL;
        err = cvector_create_vector(i + 1, 4, test_vectors[i], &vector);
        test_assert(err == CVECTOR_SUCCESS, "Vector creation");
        
        err = cvector_insert(db, vector);
        test_assert(err == CVECTOR_SUCCESS, "Vector insertion");
        
        cvector_free_vector(vector);
    }
    
    double insert_time = get_time_ms() - insert_start;
    printf("📊 INSERT PERFORMANCE: %d vectors in %.2f ms (%.2f ms/vector)\n", 
           10, insert_time, insert_time / 10);
    
    print_separator("VECTOR RETRIEVAL");
    
    // Test 12-16: Retrieve vectors
    double retrieval_start = get_time_ms();
    
    for (int i = 1; i <= 5; i++) {
        cvector_t* retrieved = NULL;
        err = cvector_get(db, i, &retrieved);
        test_assert(err == CVECTOR_SUCCESS && retrieved != NULL, "Vector retrieval");
        
        if (retrieved) {
            // Verify data integrity
            int data_correct = 1;
            for (int j = 0; j < 4; j++) {
                if (fabs(retrieved->data[j] - test_vectors[i-1][j]) > 0.0001f) {
                    data_correct = 0;
                    break;
                }
            }
            test_assert(data_correct, "Vector data integrity");
            cvector_free_vector(retrieved);
        }
    }
    
    double retrieval_time = get_time_ms() - retrieval_start;
    printf("📊 RETRIEVAL PERFORMANCE: 5 retrievals in %.2f ms (%.2f ms/retrieval)\n", 
           retrieval_time, retrieval_time / 5);
    
    print_separator("SIMILARITY SEARCH - COSINE");
    
    // Test 17-19: Cosine similarity search
    double search_start = get_time_ms();
    
    cvector_query_t query = {0};
    query.query_vector = test_vectors[0]; // Search for vector similar to [1,0,0,0]
    query.dimension = 4;
    query.top_k = 5;
    query.similarity = CVECTOR_SIMILARITY_COSINE;
    query.min_similarity = 0.0f;
    
    cvector_result_t* results = NULL;
    size_t result_count = 0;
    
    err = cvector_search(db, &query, &results, &result_count);
    test_assert(err == CVECTOR_SUCCESS, "Cosine similarity search");
    test_assert(result_count > 0, "Search returns results");
    
    if (results && result_count > 0) {
        printf("🔍 COSINE SEARCH RESULTS for [1,0,0,0]:\n");
        for (size_t i = 0; i < result_count; i++) {
            printf("   Rank %zu: ID=%llu, Similarity=%.6f\n", 
                   i+1, results[i].id, results[i].similarity);
        }
        
        // Verify ranking is correct (should find ID 1 first, then similar vectors)
        test_assert(results[0].id == 1, "Most similar vector found first");
        test_assert(results[0].similarity > 0.9f, "High similarity for exact match");
        
        cvector_free_results(results, result_count);
    }
    
    print_separator("SIMILARITY SEARCH - DOT PRODUCT");
    
    // Test 20: Dot product search
    query.similarity = CVECTOR_SIMILARITY_DOT_PRODUCT;
    err = cvector_search(db, &query, &results, &result_count);
    test_assert(err == CVECTOR_SUCCESS, "Dot product similarity search");
    
    if (results && result_count > 0) {
        printf("🔍 DOT PRODUCT SEARCH RESULTS for [1,0,0,0]:\n");
        for (size_t i = 0; i < result_count; i++) {
            printf("   Rank %zu: ID=%llu, Similarity=%.6f\n", 
                   i+1, results[i].id, results[i].similarity);
        }
        cvector_free_results(results, result_count);
    }
    
    print_separator("SIMILARITY SEARCH - EUCLIDEAN");
    
    // Test 21: Euclidean distance search
    query.similarity = CVECTOR_SIMILARITY_EUCLIDEAN;
    err = cvector_search(db, &query, &results, &result_count);
    test_assert(err == CVECTOR_SUCCESS, "Euclidean distance search");
    
    if (results && result_count > 0) {
        printf("🔍 EUCLIDEAN SEARCH RESULTS for [1,0,0,0]:\n");
        for (size_t i = 0; i < result_count; i++) {
            printf("   Rank %zu: ID=%llu, Distance=%.6f\n", 
                   i+1, results[i].id, -results[i].similarity); // Convert back to positive distance
        }
        cvector_free_results(results, result_count);
    }
    
    double search_time = get_time_ms() - search_start;
    printf("📊 SEARCH PERFORMANCE: 3 searches in %.2f ms (%.2f ms/search)\n", 
           search_time, search_time / 3);
    
    print_separator("VECTOR DELETION");
    
    // Test 22-23: Delete vectors
    double delete_start = get_time_ms();
    
    err = cvector_delete(db, 10); // Delete zero vector
    test_assert(err == CVECTOR_SUCCESS, "Vector deletion");
    
    // Verify deletion
    cvector_t* deleted_vector = NULL;
    err = cvector_get(db, 10, &deleted_vector);
    test_assert(err == CVECTOR_ERROR_VECTOR_NOT_FOUND, "Deleted vector not found");
    
    double delete_time = get_time_ms() - delete_start;
    printf("📊 DELETE PERFORMANCE: 1 deletion in %.2f ms\n", delete_time);
    
    print_separator("DATABASE STATISTICS");
    
    // Test 24: Database statistics
    cvector_db_stats_t stats = {0};
    err = cvector_db_stats(db, &stats);
    test_assert(err == CVECTOR_SUCCESS, "Database statistics");
    
    printf("📈 DATABASE STATS:\n");
    printf("   Total Vectors: %zu\n", stats.total_vectors);
    printf("   Dimension: %u\n", stats.dimension);
    printf("   File Size: %zu bytes (%.2f KB)\n", stats.total_size_bytes, stats.total_size_bytes / 1024.0);
    printf("   Default Similarity: %d\n", stats.default_similarity);
    printf("   Database Path: %s\n", stats.db_path);
    
    test_assert(stats.total_vectors == 9, "Correct vector count after deletion"); // 10 - 1 deleted
    test_assert(stats.dimension == 4, "Correct dimension");
    test_assert(stats.total_size_bytes > 0, "Non-zero file size");
    
    print_separator("DATABASE PERSISTENCE");
    
    // Test 25-26: Database persistence (close and reopen)
    err = cvector_db_close(db);
    test_assert(err == CVECTOR_SUCCESS, "Database close");
    
    // Reopen database
    cvector_db_t* reopened_db = NULL;
    err = cvector_db_open(db_path, &reopened_db);
    test_assert(err == CVECTOR_SUCCESS, "Database reopen");
    
    if (reopened_db) {
        // Test retrieval after reopen
        cvector_t* persistent_vector = NULL;
        err = cvector_get(reopened_db, 1, &persistent_vector);
        test_assert(err == CVECTOR_SUCCESS, "Vector persistence after reopen");
        
        if (persistent_vector) {
            cvector_free_vector(persistent_vector);
        }
        
        // Test search after reopen
        err = cvector_search(reopened_db, &query, &results, &result_count);
        test_assert(err == CVECTOR_SUCCESS, "Search after reopen");
        
        if (results) {
            cvector_free_results(results, result_count);
        }
        
        cvector_db_close(reopened_db);
    }
    
    print_separator("EDGE CASES");
    
    // Test 27-29: Edge cases
    cvector_db_t* edge_db = NULL;
    err = cvector_db_open(db_path, &edge_db);
    
    if (edge_db) {
        // Test non-existent vector retrieval
        cvector_t* nonexistent = NULL;
        err = cvector_get(edge_db, 999, &nonexistent);
        test_assert(err == CVECTOR_ERROR_VECTOR_NOT_FOUND, "Non-existent vector handling");
        
        // Test empty search
        query.min_similarity = 2.0f; // Impossible threshold
        err = cvector_search(edge_db, &query, &results, &result_count);
        test_assert(err == CVECTOR_SUCCESS && result_count == 0, "Empty search results");
        
        if (results) {
            cvector_free_results(results, result_count);
        }
        
        // Test invalid vector deletion
        err = cvector_delete(edge_db, 999);
        test_assert(err == CVECTOR_ERROR_VECTOR_NOT_FOUND, "Invalid deletion handling");
        
        cvector_db_close(edge_db);
    }
    
    double total_time = get_time_ms() - start_time;
    global_stats.total_time_ms = total_time;
    
    print_separator("FINAL RESULTS");
    
    printf("🏁 TEST SUMMARY:\n");
    printf("   ✅ Tests Passed: %d\n", global_stats.passed);
    printf("   ❌ Tests Failed: %d\n", global_stats.failed);
    printf("   📊 Success Rate: %.1f%%\n", 
           (float)global_stats.passed / (global_stats.passed + global_stats.failed) * 100);
    printf("   ⏱️  Total Time: %.2f ms\n", total_time);
    printf("   🗄️  Database File: %s\n", db_path);
    
    // Performance summary
    printf("\n🚀 PERFORMANCE SUMMARY:\n");
    printf("   Vector Insertion: %.2f ms/vector\n", insert_time / 10);
    printf("   Vector Retrieval: %.2f ms/retrieval\n", retrieval_time / 5);
    printf("   Similarity Search: %.2f ms/search\n", search_time / 3);
    printf("   Vector Deletion: %.2f ms/deletion\n", delete_time);
    
    // Cleanup
    remove(db_path);
    
    if (global_stats.failed == 0) {
        printf("\n🎉 ALL TESTS PASSED! The vector database is fully functional.\n");
        return 0;
    } else {
        printf("\n⚠️  Some tests failed. Check the output above for details.\n");
        return 1;
    }
}