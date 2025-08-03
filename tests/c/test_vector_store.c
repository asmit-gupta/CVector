#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include "core/cvector.h"

#define TEST_DB_PATH "./test_db.cvdb"
#define TEST_DIMENSION 128

// Test result tracking
static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) \
    do { \
        printf("Running test: %s... ", name); \
        tests_run++; \
        if (test_##name()) { \
            printf("PASSED\n"); \
            tests_passed++; \
        } else { \
            printf("FAILED\n"); \
        } \
    } while(0)

// Helper function to create test vector
cvector_t* create_test_vector(cvector_id_t id, uint32_t dimension) {
    cvector_t* vector;
    float* data = malloc(dimension * sizeof(float));
    
    // Fill with test data
    for (uint32_t i = 0; i < dimension; i++) {
        data[i] = (float)(id * dimension + i) / 1000.0f;
    }
    
    cvector_error_t err = cvector_create_vector(id, dimension, data, &vector);
    free(data);
    
    return (err == CVECTOR_SUCCESS) ? vector : NULL;
}

// Clean up test database
void cleanup_test_db() {
    unlink(TEST_DB_PATH);
}

// Test database creation
int test_db_create() {
    cleanup_test_db();
    
    cvector_db_config_t config = {0};
    strncpy(config.name, "test_db", sizeof(config.name) - 1);
    strncpy(config.data_path, TEST_DB_PATH, sizeof(config.data_path) - 1);
    config.dimension = TEST_DIMENSION;
    config.default_similarity = CVECTOR_SIMILARITY_COSINE;
    config.memory_mapped = false;
    config.max_vectors = 1000;
    
    cvector_db_t* db;
    cvector_error_t err = cvector_db_create(&config, &db);
    
    if (err != CVECTOR_SUCCESS) {
        return 0;
    }
    
    // Verify database was created
    cvector_db_stats_t stats;
    err = cvector_db_stats(db, &stats);
    
    int success = (err == CVECTOR_SUCCESS) && 
                  (stats.dimension == TEST_DIMENSION) &&
                  (stats.total_vectors == 0);
    
    cvector_db_close(db);
    cleanup_test_db();
    
    return success;
}

// Test database open
int test_db_open() {
    cleanup_test_db();
    
    // First create a database
    cvector_db_config_t config = {0};
    strncpy(config.name, "test_db", sizeof(config.name) - 1);
    strncpy(config.data_path, TEST_DB_PATH, sizeof(config.data_path) - 1);
    config.dimension = TEST_DIMENSION;
    config.default_similarity = CVECTOR_SIMILARITY_COSINE;
    config.memory_mapped = false;
    config.max_vectors = 1000;
    
    cvector_db_t* db;
    cvector_error_t err = cvector_db_create(&config, &db);
    if (err != CVECTOR_SUCCESS) {
        return 0;
    }
    cvector_db_close(db);
    
    // Now try to open it
    err = cvector_db_open(TEST_DB_PATH, &db);
    if (err != CVECTOR_SUCCESS) {
        cleanup_test_db();
        return 0;
    }
    
    // Verify it opened correctly
    cvector_db_stats_t stats;
    err = cvector_db_stats(db, &stats);
    
    int success = (err == CVECTOR_SUCCESS) && 
                  (stats.dimension == TEST_DIMENSION);
    
    cvector_db_close(db);
    cleanup_test_db();
    
    return success;
}

// Test vector insertion
int test_vector_insert() {
    cleanup_test_db();
    
    cvector_db_config_t config = {0};
    strncpy(config.name, "test_db", sizeof(config.name) - 1);
    strncpy(config.data_path, TEST_DB_PATH, sizeof(config.data_path) - 1);
    config.dimension = TEST_DIMENSION;
    config.default_similarity = CVECTOR_SIMILARITY_COSINE;
    config.memory_mapped = false;
    config.max_vectors = 1000;
    
    cvector_db_t* db;
    cvector_error_t err = cvector_db_create(&config, &db);
    if (err != CVECTOR_SUCCESS) {
        return 0;
    }
    
    // Create and insert test vector
    cvector_t* vector = create_test_vector(1, TEST_DIMENSION);
    if (!vector) {
        cvector_db_close(db);
        cleanup_test_db();
        return 0;
    }
    
    err = cvector_insert(db, vector);
    int success = (err == CVECTOR_SUCCESS);
    
    if (success) {
        // Verify stats updated
        cvector_db_stats_t stats;
        err = cvector_db_stats(db, &stats);
        success = (err == CVECTOR_SUCCESS) && (stats.total_vectors == 1);
    }
    
    cvector_free_vector(vector);
    cvector_db_close(db);
    cleanup_test_db();
    
    return success;
}

// Test vector retrieval
int test_vector_get() {
    cleanup_test_db();
    
    cvector_db_config_t config = {0};
    strncpy(config.name, "test_db", sizeof(config.name) - 1);
    strncpy(config.data_path, TEST_DB_PATH, sizeof(config.data_path) - 1);
    config.dimension = TEST_DIMENSION;
    config.default_similarity = CVECTOR_SIMILARITY_COSINE;
    config.memory_mapped = false;
    config.max_vectors = 1000;
    
    cvector_db_t* db;
    cvector_error_t err = cvector_db_create(&config, &db);
    if (err != CVECTOR_SUCCESS) {
        return 0;
    }
    
    // Insert test vector
    cvector_t* original = create_test_vector(42, TEST_DIMENSION);
    if (!original) {
        cvector_db_close(db);
        cleanup_test_db();
        return 0;
    }
    
    err = cvector_insert(db, original);
    if (err != CVECTOR_SUCCESS) {
        cvector_free_vector(original);
        cvector_db_close(db);
        cleanup_test_db();
        return 0;
    }
    
    // Retrieve vector
    cvector_t* retrieved;
    err = cvector_get(db, 42, &retrieved);
    if (err != CVECTOR_SUCCESS) {
        cvector_free_vector(original);
        cvector_db_close(db);
        cleanup_test_db();
        return 0;
    }
    
    // Verify data matches
    int success = (retrieved->id == original->id) &&
                  (retrieved->dimension == original->dimension);
    
    if (success) {
        // Check vector data
        for (uint32_t i = 0; i < original->dimension && success; i++) {
            if (retrieved->data[i] != original->data[i]) {
                success = 0;
            }
        }
    }
    
    cvector_free_vector(original);
    cvector_free_vector(retrieved);
    cvector_db_close(db);
    cleanup_test_db();
    
    return success;
}

// Test vector deletion
int test_vector_delete() {
    cleanup_test_db();
    
    cvector_db_config_t config = {0};
    strncpy(config.name, "test_db", sizeof(config.name) - 1);
    strncpy(config.data_path, TEST_DB_PATH, sizeof(config.data_path) - 1);
    config.dimension = TEST_DIMENSION;
    config.default_similarity = CVECTOR_SIMILARITY_COSINE;
    config.memory_mapped = false;
    config.max_vectors = 1000;
    
    cvector_db_t* db;
    cvector_error_t err = cvector_db_create(&config, &db);
    if (err != CVECTOR_SUCCESS) {
        return 0;
    }
    
    // Insert test vector
    cvector_t* vector = create_test_vector(100, TEST_DIMENSION);
    if (!vector) {
        cvector_db_close(db);
        cleanup_test_db();
        return 0;
    }
    
    err = cvector_insert(db, vector);
    if (err != CVECTOR_SUCCESS) {
        cvector_free_vector(vector);
        cvector_db_close(db);
        cleanup_test_db();
        return 0;
    }
    
    // Delete vector
    err = cvector_delete(db, 100);
    int success = (err == CVECTOR_SUCCESS);
    
    if (success) {
        // Verify it's gone
        cvector_t* retrieved;
        err = cvector_get(db, 100, &retrieved);
        success = (err == CVECTOR_ERROR_VECTOR_NOT_FOUND);
    }
    
    cvector_free_vector(vector);
    cvector_db_close(db);
    cleanup_test_db();
    
    return success;
}

// Test multiple vector operations
int test_multiple_vectors() {
    cleanup_test_db();
    
    cvector_db_config_t config = {0};
    strncpy(config.name, "test_db", sizeof(config.name) - 1);
    strncpy(config.data_path, TEST_DB_PATH, sizeof(config.data_path) - 1);
    config.dimension = TEST_DIMENSION;
    config.default_similarity = CVECTOR_SIMILARITY_COSINE;
    config.memory_mapped = false;
    config.max_vectors = 1000;
    
    cvector_db_t* db;
    cvector_error_t err = cvector_db_create(&config, &db);
    if (err != CVECTOR_SUCCESS) {
        return 0;
    }
    
    const int num_vectors = 10;
    int success = 1;
    
    // Insert multiple vectors
    for (int i = 1; i <= num_vectors && success; i++) {
        cvector_t* vector = create_test_vector(i, TEST_DIMENSION);
        if (!vector) {
            success = 0;
            break;
        }
        
        err = cvector_insert(db, vector);
        if (err != CVECTOR_SUCCESS) {
            success = 0;
        }
        
        cvector_free_vector(vector);
    }
    
    // Verify all vectors can be retrieved
    for (int i = 1; i <= num_vectors && success; i++) {
        cvector_t* retrieved;
        err = cvector_get(db, i, &retrieved);
        if (err != CVECTOR_SUCCESS) {
            success = 0;
        } else {
            if (retrieved->id != (cvector_id_t)i) {
                success = 0;
            }
            cvector_free_vector(retrieved);
        }
    }
    
    // Check stats
    if (success) {
        cvector_db_stats_t stats;
        err = cvector_db_stats(db, &stats);
        success = (err == CVECTOR_SUCCESS) && 
                  (stats.total_vectors == num_vectors);
    }
    
    cvector_db_close(db);
    cleanup_test_db();
    
    return success;
}

// Test error conditions
int test_error_conditions() {
    cleanup_test_db();
    
    // Test opening non-existent database
    cvector_db_t* db;
    cvector_error_t err = cvector_db_open("non_existent.cvdb", &db);
    if (err != CVECTOR_ERROR_DB_NOT_FOUND) {
        return 0;
    }
    
    // Test invalid config
    cvector_db_config_t config = {0};
    config.dimension = 0; // Invalid
    err = cvector_db_create(&config, &db);
    if (err != CVECTOR_ERROR_INVALID_ARGS) {
        return 0;
    }
    
    // Test null arguments
    err = cvector_db_create(NULL, &db);
    if (err != CVECTOR_ERROR_INVALID_ARGS) {
        return 0;
    }
    
    return 1;
}

int main() {
    printf("Running CVector C Tests\n");
    printf("=====================\n\n");
    
    TEST(test_db_create);
    TEST(test_db_open);
    TEST(test_vector_insert);
    TEST(test_vector_get);
    TEST(test_vector_delete);
    TEST(test_multiple_vectors);
    TEST(test_error_conditions);
    
    printf("\nTest Results:\n");
    printf("=============\n");
    printf("Tests run: %d\n", tests_run);
    printf("Tests passed: %d\n", tests_passed);
    printf("Tests failed: %d\n", tests_run - tests_passed);
    
    if (tests_passed == tests_run) {
        printf("\nAll tests PASSED! ✅\n");
        return 0;
    } else {
        printf("\nSome tests FAILED! ❌\n");
        return 1;
    }
}