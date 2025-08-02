package cvector

/*
#cgo CFLAGS: -I../../src
#cgo LDFLAGS: -L../../build -lcvector
#include "core/cvector.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"runtime"
	"time"
	"unsafe"
)

// Error wraps CVector error codes
type Error int

const (
	Success              Error = C.CVECTOR_SUCCESS
	ErrInvalidArgs       Error = C.CVECTOR_ERROR_INVALID_ARGS
	ErrOutOfMemory       Error = C.CVECTOR_ERROR_OUT_OF_MEMORY
	ErrFileIO            Error = C.CVECTOR_ERROR_FILE_IO
	ErrDBNotFound        Error = C.CVECTOR_ERROR_DB_NOT_FOUND
	ErrVectorNotFound    Error = C.CVECTOR_ERROR_VECTOR_NOT_FOUND
	ErrDimensionMismatch Error = C.CVECTOR_ERROR_DIMENSION_MISMATCH
	ErrDBCorrupt         Error = C.CVECTOR_ERROR_DB_CORRUPT
)

func (e Error) Error() string {
	return C.GoString(C.cvector_error_string(C.cvector_error_t(e)))
}

// SimilarityType represents different similarity metrics
type SimilarityType int

const (
	SimilarityCosine      SimilarityType = C.CVECTOR_SIMILARITY_COSINE
	SimilarityDotProduct  SimilarityType = C.CVECTOR_SIMILARITY_DOT_PRODUCT
	SimilarityEuclidean   SimilarityType = C.CVECTOR_SIMILARITY_EUCLIDEAN
)

// DBConfig holds database configuration
type DBConfig struct {
	Name              string
	DataPath          string
	Dimension         uint32
	DefaultSimilarity SimilarityType
	MemoryMapped      bool
	MaxVectors        int
}

// DB represents a CVector database
type DB struct {
	db *C.cvector_db_t
}

// Vector represents a vector with metadata
type Vector struct {
	ID        uint64
	Dimension uint32
	Data      []float32
	Timestamp time.Time
}

// Result represents a search result
type Result struct {
	ID         uint64
	Similarity float32
	Vector     *Vector // Optional: full vector data
}

// Query represents a search query
type Query struct {
	QueryVector   []float32
	TopK          uint32
	Similarity    SimilarityType
	MinSimilarity float32
}

// Stats holds database statistics
type Stats struct {
	TotalVectors      int
	TotalSizeBytes    int
	Dimension         uint32
	DefaultSimilarity SimilarityType
	DBPath            string
}

// CreateDB creates a new vector database
func CreateDB(config *DBConfig) (*DB, error) {
	if config == nil {
		return nil, Error(ErrInvalidArgs)
	}

	// Convert Go config to C config
	cConfig := C.cvector_db_config_t{}
	
	nameBytes := []byte(config.Name)
	if len(nameBytes) >= C.CVECTOR_MAX_DB_NAME {
		return nil, Error(ErrInvalidArgs)
	}
	copy((*[C.CVECTOR_MAX_DB_NAME]C.char)(unsafe.Pointer(&cConfig.name[0]))[:], nameBytes)
	
	pathBytes := []byte(config.DataPath)
	if len(pathBytes) >= C.CVECTOR_MAX_PATH {
		return nil, Error(ErrInvalidArgs)
	}
	copy((*[C.CVECTOR_MAX_PATH]C.char)(unsafe.Pointer(&cConfig.data_path[0]))[:], pathBytes)
	
	cConfig.dimension = C.uint32_t(config.Dimension)
	cConfig.default_similarity = C.cvector_similarity_t(config.DefaultSimilarity)
	cConfig.memory_mapped = C.bool(config.MemoryMapped)
	cConfig.max_vectors = C.size_t(config.MaxVectors)

	var cDB *C.cvector_db_t
	result := C.cvector_db_create(&cConfig, &cDB)
	if result != C.CVECTOR_SUCCESS {
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
	if result != C.CVECTOR_SUCCESS {
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
	
	if result != C.CVECTOR_SUCCESS {
		return Error(result)
	}
	return nil
}

// DropDB removes a database file
func DropDB(dbPath string) error {
	cPath := C.CString(dbPath)
	defer C.free(unsafe.Pointer(cPath))

	result := C.cvector_db_drop(cPath)
	if result != C.CVECTOR_SUCCESS {
		return Error(result)
	}
	return nil
}

// Insert adds a vector to the database
func (db *DB) Insert(vector *Vector) error {
	if db.db == nil {
		return Error(ErrInvalidArgs)
	}
	if vector == nil || len(vector.Data) == 0 {
		return Error(ErrInvalidArgs)
	}

	// Convert Go vector to C vector
	cVector := C.cvector_t{
		id:        C.cvector_id_t(vector.ID),
		dimension: C.uint32_t(vector.Dimension),
		timestamp: C.uint64_t(vector.Timestamp.Unix()),
	}

	// Allocate C array for vector data
	dataSize := len(vector.Data) * int(unsafe.Sizeof(float32(0)))
	cVector.data = (*C.float)(C.malloc(C.size_t(dataSize)))
	if cVector.data == nil {
		return Error(ErrOutOfMemory)
	}
	defer C.free(unsafe.Pointer(cVector.data))

	// Copy data
	cDataSlice := (*[1 << 20]C.float)(unsafe.Pointer(cVector.data))[:len(vector.Data):len(vector.Data)]
	for i, v := range vector.Data {
		cDataSlice[i] = C.float(v)
	}

	result := C.cvector_insert(db.db, &cVector)
	if result != C.CVECTOR_SUCCESS {
		return Error(result)
	}
	return nil
}

// Get retrieves a vector by ID
func (db *DB) Get(id uint64) (*Vector, error) {
	if db.db == nil {
		return nil, Error(ErrInvalidArgs)
	}

	var cVector *C.cvector_t
	result := C.cvector_get(db.db, C.cvector_id_t(id), &cVector)
	if result != C.CVECTOR_SUCCESS {
		return nil, Error(result)
	}
	defer C.cvector_free_vector(cVector)

	// Convert C vector to Go vector
	vector := &Vector{
		ID:        uint64(cVector.id),
		Dimension: uint32(cVector.dimension),
		Timestamp: time.Unix(int64(cVector.timestamp), 0),
	}

	// Copy vector data
	dataSize := int(cVector.dimension)
	vector.Data = make([]float32, dataSize)
	cDataSlice := (*[1 << 20]C.float)(unsafe.Pointer(cVector.data))[:dataSize:dataSize]
	for i, v := range cDataSlice {
		vector.Data[i] = float32(v)
	}

	return vector, nil
}

// Delete removes a vector by ID
func (db *DB) Delete(id uint64) error {
	if db.db == nil {
		return Error(ErrInvalidArgs)
	}

	result := C.cvector_delete(db.db, C.cvector_id_t(id))
	if result != C.CVECTOR_SUCCESS {
		return Error(result)
	}
	return nil
}

// Stats returns database statistics
func (db *DB) Stats() (*Stats, error) {
	if db.db == nil {
		return nil, Error(ErrInvalidArgs)
	}

	var cStats C.cvector_db_stats_t
	result := C.cvector_db_stats(db.db, &cStats)
	if result != C.CVECTOR_SUCCESS {
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

// NewVector creates a new vector with the current timestamp
func NewVector(id uint64, data []float32) *Vector {
	return &Vector{
		ID:        id,
		Dimension: uint32(len(data)),
		Data:      data,
		Timestamp: time.Now(),
	}
}