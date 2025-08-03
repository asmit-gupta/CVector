#include "similarity.h"
#include <math.h>
#include <float.h>

float cvector_cosine_similarity(const float* a, const float* b, uint32_t dimension) {
    if (!a || !b || dimension == 0) {
        return 0.0f;
    }
    
    float dot_product = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;
    
    for (uint32_t i = 0; i < dimension; i++) {
        dot_product += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    
    norm_a = sqrtf(norm_a);
    norm_b = sqrtf(norm_b);
    
    if (norm_a < FLT_EPSILON || norm_b < FLT_EPSILON) {
        return 0.0f;
    }
    
    return dot_product / (norm_a * norm_b);
}

float cvector_dot_product(const float* a, const float* b, uint32_t dimension) {
    if (!a || !b || dimension == 0) {
        return 0.0f;
    }
    
    float dot_product = 0.0f;
    for (uint32_t i = 0; i < dimension; i++) {
        dot_product += a[i] * b[i];
    }
    
    return dot_product;
}

float cvector_euclidean_distance(const float* a, const float* b, uint32_t dimension) {
    if (!a || !b || dimension == 0) {
        return 0.0f;
    }
    
    float sum_squared_diff = 0.0f;
    for (uint32_t i = 0; i < dimension; i++) {
        float diff = a[i] - b[i];
        sum_squared_diff += diff * diff;
    }
    
    return sqrtf(sum_squared_diff);
}

float cvector_vector_norm(const float* vector, uint32_t dimension) {
    if (!vector || dimension == 0) {
        return 0.0f;
    }
    
    float sum_squared = 0.0f;
    for (uint32_t i = 0; i < dimension; i++) {
        sum_squared += vector[i] * vector[i];
    }
    
    return sqrtf(sum_squared);
}

void cvector_normalize_vector(float* vector, uint32_t dimension) {
    if (!vector || dimension == 0) {
        return;
    }
    
    float norm = cvector_vector_norm(vector, dimension);
    if (norm < FLT_EPSILON) {
        return; // Cannot normalize zero vector
    }
    
    for (uint32_t i = 0; i < dimension; i++) {
        vector[i] /= norm;
    }
}