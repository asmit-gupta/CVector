package cvector

/*
#cgo CFLAGS: -I../../src -std=c11
#cgo LDFLAGS: -L../../build -lcvector -lm

#include "core/cvector.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Wrapper functions to avoid CGO struct issues
cvector_error_t create_db_wrapper(const char* name, const char* path, uint32_t dimension, cvector_db_t** db) {
    cvector_db_config_t config = {0};

    strncpy(config.name, name, CVECTOR_MAX_DB_NAME - 1);
    strncpy(config.data_path, path, CVECTOR_MAX_PATH - 1);
    config.dimension = dimension;
    config.default_similarity = CVECTOR_SIMILARITY_COSINE;
    config.memory_mapped = false;
    config.max_vectors = 1000000;

    return cvector_db_create(&config, db);
}

cvector_error_t insert_vector_wrapper(cvector_db_t* db, uint64_t id, uint32_t dimension, float* data) {
    cvector_t vector = {0};
    vector.id = id;
    vector.dimension = dimension;
    vector.data = data;
    vector.timestamp = (uint64_t)time(NULL);

    return cvector_insert(db, &vector);
}

cvector_error_t search_wrapper(cvector_db_t* db, float* query_vector, uint32_t dimension, 
                              uint32_t top_k, cvector_similarity_t similarity, float min_similarity,
                              cvector_result_t** results, size_t* result_count) {
    cvector_query_t query = {0};
    query.query_vector = query_vector;
    query.dimension = dimension;
    query.top_k = top_k;
    query.similarity = similarity;
    query.min_similarity = min_similarity;
    
    return cvector_search(db, &query, results, result_count);
}
*/
import "C"
import (
	"runtime"
	"time"
	"unsafe"
)

// DB represents a CVector database
type DB struct {
	db *C.cvector_db_t
}

// CreateDB creates a new vector database
func CreateDB(config *DBConfig) (*DB, error) {
	if config == nil {
		return nil, ErrInvalidArgs
	}

	cName := C.CString(config.Name)
	defer C.free(unsafe.Pointer(cName))
	
	cPath := C.CString(config.DataPath)
	defer C.free(unsafe.Pointer(cPath))

	var cDB *C.cvector_db_t
	result := C.create_db_wrapper(cName, cPath, C.uint32_t(config.Dimension), &cDB)
	
	if result != 0 {
		return nil, Error(result)
	}

	db := &DB{db: cDB}
	runtime.SetFinalizer(db, (*DB).Close)
	
	return db, nil
}

// OpenDB opens an existing vector database
func OpenDB(dbPath string) (*DB, error) {
	cPath := C.CString(dbPath)
	defer C.free(unsafe.Pointer(cPath))

	var cDB *C.cvector_db_t
	result := C.cvector_db_open(cPath, &cDB)
	if result != 0 {
		return nil, Error(result)
	}

	db := &DB{db: cDB}
	runtime.SetFinalizer(db, (*DB).Close)
	
	return db, nil
}

// Close closes the database
func (db *DB) Close() error {
	if db.db == nil {
		return nil
	}

	result := C.cvector_db_close(db.db)
	db.db = nil
	runtime.SetFinalizer(db, nil)
	
	if result != 0 {
		return Error(result)
	}
	return nil
}

// DropDB removes a database file
func DropDB(dbPath string) error {
	cPath := C.CString(dbPath)
	defer C.free(unsafe.Pointer(cPath))

	result := C.cvector_db_drop(cPath)
	if result != 0 {
		return Error(result)
	}
	return nil
}

// Insert adds a vector to the database
func (db *DB) Insert(vector *Vector) error {
	if db.db == nil {
		return ErrInvalidArgs
	}
	if vector == nil || len(vector.Data) == 0 {
		return ErrInvalidArgs
	}

	// Allocate C array for vector data
	dataSize := len(vector.Data)
	cData := (*C.float)(C.malloc(C.size_t(dataSize * 4))) // 4 bytes per float32
	if cData == nil {
		return ErrOutOfMemory
	}
	defer C.free(unsafe.Pointer(cData))

	// Copy data using slice header manipulation  
	cDataSlice := (*[1 << 20]C.float)(unsafe.Pointer(cData))[:dataSize:dataSize]
	for i, v := range vector.Data {
		cDataSlice[i] = C.float(v)
	}

	// Use wrapper function instead of creating struct in Go
	result := C.insert_vector_wrapper(db.db, C.uint64_t(vector.ID), C.uint32_t(vector.Dimension), cData)
	if result != 0 {
		return Error(result)
	}
	return nil
}

// Get retrieves a vector by ID
func (db *DB) Get(id uint64) (*Vector, error) {
	if db.db == nil {
		return nil, ErrInvalidArgs
	}

	var cVector *C.cvector_t
	result := C.cvector_get(db.db, C.cvector_id_t(id), &cVector)
	if result != 0 {
		return nil, Error(result)
	}
	defer C.cvector_free_vector(cVector)

	// Convert C vector to Go vector
	vector := &Vector{
		ID:        uint64(cVector.id),
		Dimension: uint32(cVector.dimension),
		Timestamp: time.Unix(int64(cVector.timestamp), 0),
	}

	// Copy vector data safely
	dataSize := int(cVector.dimension)
	vector.Data = make([]float32, dataSize)
	if cVector.data != nil {
		cDataSlice := (*[1 << 20]C.float)(unsafe.Pointer(cVector.data))[:dataSize:dataSize]
		for i, v := range cDataSlice {
			vector.Data[i] = float32(v)
		}
	}

	return vector, nil
}

// Delete removes a vector by ID
func (db *DB) Delete(id uint64) error {
	if db.db == nil {
		return ErrInvalidArgs
	}

	result := C.cvector_delete(db.db, C.cvector_id_t(id))
	if result != 0 {
		return Error(result)
	}
	return nil
}

// Stats returns database statistics
func (db *DB) Stats() (*Stats, error) {
	if db.db == nil {
		return nil, ErrInvalidArgs
	}

	var cStats C.cvector_db_stats_t
	result := C.cvector_db_stats(db.db, &cStats)
	if result != 0 {
		return nil, Error(result)
	}

	stats := &Stats{
		TotalVectors:      int(cStats.total_vectors),
		TotalSizeBytes:    int(cStats.total_size_bytes),
		Dimension:         uint32(cStats.dimension),
		DefaultSimilarity: SimilarityType(cStats.default_similarity),
		DBPath:            C.GoString(&cStats.db_path[0]),
	}

	return stats, nil
}

// Search performs a similarity search on the database
func (db *DB) Search(query *Query) ([]*Result, error) {
	if db.db == nil {
		return nil, ErrInvalidArgs
	}
	if query == nil || len(query.QueryVector) == 0 {
		return nil, ErrInvalidArgs
	}

	// Allocate C array for query vector
	dataSize := len(query.QueryVector)
	cData := (*C.float)(C.malloc(C.size_t(dataSize * 4))) // 4 bytes per float32
	if cData == nil {
		return nil, ErrOutOfMemory
	}
	defer C.free(unsafe.Pointer(cData))

	// Copy query vector data
	cDataSlice := (*[1 << 20]C.float)(unsafe.Pointer(cData))[:dataSize:dataSize]
	for i, v := range query.QueryVector {
		cDataSlice[i] = C.float(v)
	}

	// Perform search
	var cResults *C.cvector_result_t
	var resultCount C.size_t
	
	result := C.search_wrapper(
		db.db,
		cData,
		C.uint32_t(len(query.QueryVector)),
		C.uint32_t(query.TopK),
		C.cvector_similarity_t(query.Similarity),
		C.float(query.MinSimilarity),
		&cResults,
		&resultCount,
	)
	
	if result != 0 {
		return nil, Error(result)
	}

	if resultCount == 0 || cResults == nil {
		return []*Result{}, nil
	}
	defer C.cvector_free_results(cResults, resultCount)

	// Convert C results to Go results
	results := make([]*Result, int(resultCount))
	cResultsSlice := (*[1 << 20]C.cvector_result_t)(unsafe.Pointer(cResults))[:int(resultCount):int(resultCount)]
	
	for i, cResult := range cResultsSlice {
		results[i] = &Result{
			ID:         uint64(cResult.id),
			Similarity: float32(cResult.similarity),
			Vector:     nil, // Vector data not loaded by default
		}
	}

	return results, nil
}

// NewVector creates a new vector with the current timestamp
func NewVector(id uint64, data []float32) *Vector {
	return &Vector{
		ID:        id,
		Dimension: uint32(len(data)),
		Data:      data,
		Timestamp: time.Now(),
	}
}