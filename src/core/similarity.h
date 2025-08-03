#ifndef CVECTOR_SIMILARITY_H
#define CVECTOR_SIMILARITY_H

#include <stdint.h>

// Similarity calculation functions
float cvector_cosine_similarity(const float* a, const float* b, uint32_t dimension);
float cvector_dot_product(const float* a, const float* b, uint32_t dimension);
float cvector_euclidean_distance(const float* a, const float* b, uint32_t dimension);

// Helper functions
float cvector_vector_norm(const float* vector, uint32_t dimension);
void cvector_normalize_vector(float* vector, uint32_t dimension);

#endif // CVECTOR_SIMILARITY_H