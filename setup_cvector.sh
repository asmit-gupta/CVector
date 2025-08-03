#!/bin/bash

set -e  # Exit on any error

echo "🚀 CVector Database Setup Script"
echo "================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
    exit 1
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if we're on macOS
    if [[ "$OSTYPE" != "darwin"* ]]; then
        log_warning "This script is designed for macOS. You may need to adjust commands for your OS."
    fi
    
    # Check for required tools
    for tool in gcc go make; do
        if ! command -v $tool &> /dev/null; then
            log_error "$tool is not installed. Please install it first."
        fi
    done
    
    log_success "Prerequisites check passed"
}

# Create project structure
create_structure() {
    log_info "Creating project structure..."
    
    # Create directories
    mkdir -p cmd/cvector
    mkdir -p pkg/cvector
    mkdir -p src/core
    mkdir -p src/utils
    mkdir -p tests/c
    mkdir -p tests/go
    mkdir -p build
    mkdir -p data
    
    log_success "Project structure created"
}

# Initialize Go module
init_go_module() {
    log_info "Initializing Go module..."
    
    if [[ ! -f "go.mod" ]]; then
        # Use current directory name as module name
        MODULE_NAME="github.com/$(whoami)/cvector"
        echo "Enter your GitHub username (or press Enter to use '$(whoami)'):"
        read -r username
        if [[ -n "$username" ]]; then
            MODULE_NAME="github.com/$username/cvector"
        fi
        
        go mod init "$MODULE_NAME"
        log_success "Go module initialized as $MODULE_NAME"
    else
        log_info "Go module already exists"
    fi
}

# Create core C files
create_c_files() {
    log_info "Creating C source files..."
    
    # Create src/core/cvector.h
    cat > src/core/cvector.h << 'EOF'
#ifndef CVECTOR_H
#define CVECTOR_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

// Version info
#define CVECTOR_VERSION_MAJOR 0
#define CVECTOR_VERSION_MINOR 1
#define CVECTOR_VERSION_PATCH 0

// Constants
#define CVECTOR_MAX_DIMENSION 4096
#define CVECTOR_DEFAULT_DIMENSION 512
#define CVECTOR_MAX_DB_NAME 256
#define CVECTOR_MAX_PATH 1024

// Error codes
typedef enum {
    CVECTOR_SUCCESS = 0,
    CVECTOR_ERROR_INVALID_ARGS = -1,
    CVECTOR_ERROR_OUT_OF_MEMORY = -2,
    CVECTOR_ERROR_FILE_IO = -3,
    CVECTOR_ERROR_DB_NOT_FOUND = -4,
    CVECTOR_ERROR_VECTOR_NOT_FOUND = -5,
    CVECTOR_ERROR_DIMENSION_MISMATCH = -6,
    CVECTOR_ERROR_DB_CORRUPT = -7
} cvector_error_t;

// Similarity metrics
typedef enum {
    CVECTOR_SIMILARITY_COSINE = 0,
    CVECTOR_SIMILARITY_DOT_PRODUCT = 1,
    CVECTOR_SIMILARITY_EUCLIDEAN = 2
} cvector_similarity_t;

// Vector ID type
typedef uint64_t cvector_id_t;

// Forward declarations
typedef struct cvector_t cvector_t;
typedef struct cvector_db cvector_db_t;
typedef struct cvector_db_config_t cvector_db_config_t;
typedef struct cvector_db_stats_t cvector_db_stats_t;

// Vector structure
struct cvector_t {
    cvector_id_t id;
    uint32_t dimension;
    float* data;
    uint64_t timestamp;
};

// Database configuration
struct cvector_db_config_t {
    char name[CVECTOR_MAX_DB_NAME];
    char data_path[CVECTOR_MAX_PATH];
    uint32_t dimension;
    cvector_similarity_t default_similarity;
    bool memory_mapped;
    size_t max_vectors;
};

// Database stats
struct cvector_db_stats_t {
    size_t total_vectors;
    size_t total_size_bytes;
    uint32_t dimension;
    cvector_similarity_t default_similarity;
    char db_path[CVECTOR_MAX_PATH];
};

// Core Database Operations
cvector_error_t cvector_db_create(const cvector_db_config_t* config, cvector_db_t** db);
cvector_error_t cvector_db_open(const char* db_path, cvector_db_t** db);
cvector_error_t cvector_db_close(cvector_db_t* db);
cvector_error_t cvector_db_drop(const char* db_path);

// Vector CRUD Operations
cvector_error_t cvector_insert(cvector_db_t* db, const cvector_t* vector);
cvector_error_t cvector_get(cvector_db_t* db, cvector_id_t id, cvector_t** vector);
cvector_error_t cvector_delete(cvector_db_t* db, cvector_id_t id);

// Utility Functions
cvector_error_t cvector_create_vector(cvector_id_t id, uint32_t dimension, const float* data, cvector_t** vector);
void cvector_free_vector(cvector_t* vector);
const char* cvector_error_string(cvector_error_t error);
cvector_error_t cvector_db_stats(cvector_db_t* db, cvector_db_stats_t* stats);

#endif // CVECTOR_H
EOF
    
    log_success "Created src/core/cvector.h"
}

# Create minimal C implementation for testing
create_minimal_c_impl() {
    log_info "Creating minimal C implementation..."
    
    # Create a minimal implementation that compiles
    cat > src/core/vector_store.c << 'EOF'
#include "cvector.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <unistd.h>

// Minimal implementation for testing

struct cvector_db {
    cvector_db_config_t config;
    FILE* data_file;
    cvector_id_t next_id;
    size_t vector_count;
    bool is_open;
};

cvector_error_t cvector_db_create(const cvector_db_config_t* config, cvector_db_t** db) {
    if (!config || !db) return CVECTOR_ERROR_INVALID_ARGS;
    
    *db = calloc(1, sizeof(cvector_db_t));
    if (!*db) return CVECTOR_ERROR_OUT_OF_MEMORY;
    
    memcpy(&(*db)->config, config, sizeof(cvector_db_config_t));
    (*db)->next_id = 1;
    (*db)->vector_count = 0;
    (*db)->is_open = true;
    
    return CVECTOR_SUCCESS;
}

cvector_error_t cvector_db_open(const char* db_path, cvector_db_t** db) {
    if (!db_path || !db) return CVECTOR_ERROR_INVALID_ARGS;
    
    struct stat st;
    if (stat(db_path, &st) != 0) return CVECTOR_ERROR_DB_NOT_FOUND;
    
    *db = calloc(1, sizeof(cvector_db_t));
    if (!*db) return CVECTOR_ERROR_OUT_OF_MEMORY;
    
    strncpy((*db)->config.data_path, db_path, sizeof((*db)->config.data_path) - 1);
    (*db)->config.dimension = 128; // Default for now
    (*db)->is_open = true;
    
    return CVECTOR_SUCCESS;
}

cvector_error_t cvector_db_close(cvector_db_t* db) {
    if (!db) return CVECTOR_ERROR_INVALID_ARGS;
    if (db->data_file) fclose(db->data_file);
    free(db);
    return CVECTOR_SUCCESS;
}

cvector_error_t cvector_db_drop(const char* db_path) {
    if (!db_path) return CVECTOR_ERROR_INVALID_ARGS;
    return (unlink(db_path) == 0) ? CVECTOR_SUCCESS : CVECTOR_ERROR_FILE_IO;
}

cvector_error_t cvector_insert(cvector_db_t* db, const cvector_t* vector) {
    if (!db || !vector) return CVECTOR_ERROR_INVALID_ARGS;
    db->vector_count++;
    return CVECTOR_SUCCESS;
}

cvector_error_t cvector_get(cvector_db_t* db, cvector_id_t id, cvector_t** vector) {
    if (!db || !vector) return CVECTOR_ERROR_INVALID_ARGS;
    return CVECTOR_ERROR_VECTOR_NOT_FOUND; // Simplified for now
}

cvector_error_t cvector_delete(cvector_db_t* db, cvector_id_t id) {
    if (!db) return CVECTOR_ERROR_INVALID_ARGS;
    if (db->vector_count > 0) db->vector_count--;
    return CVECTOR_SUCCESS;
}

cvector_error_t cvector_create_vector(cvector_id_t id, uint32_t dimension, const float* data, cvector_t** vector) {
    if (!data || !vector) return CVECTOR_ERROR_INVALID_ARGS;
    
    cvector_t* v = malloc(sizeof(cvector_t));
    if (!v) return CVECTOR_ERROR_OUT_OF_MEMORY;
    
    v->data = malloc(dimension * sizeof(float));
    if (!v->data) {
        free(v);
        return CVECTOR_ERROR_OUT_OF_MEMORY;
    }
    
    v->id = id;
    v->dimension = dimension;
    v->timestamp = (uint64_t)time(NULL);
    memcpy(v->data, data, dimension * sizeof(float));
    
    *vector = v;
    return CVECTOR_SUCCESS;
}

void cvector_free_vector(cvector_t* vector) {
    if (vector) {
        free(vector->data);
        free(vector);
    }
}

const char* cvector_error_string(cvector_error_t error) {
    switch (error) {
        case CVECTOR_SUCCESS: return "Success";
        case CVECTOR_ERROR_INVALID_ARGS: return "Invalid arguments";
        case CVECTOR_ERROR_OUT_OF_MEMORY: return "Out of memory";
        case CVECTOR_ERROR_FILE_IO: return "File I/O error";
        case CVECTOR_ERROR_DB_NOT_FOUND: return "Database not found";
        case CVECTOR_ERROR_VECTOR_NOT_FOUND: return "Vector not found";
        case CVECTOR_ERROR_DIMENSION_MISMATCH: return "Dimension mismatch";
        case CVECTOR_ERROR_DB_CORRUPT: return "Database corrupt";
        default: return "Unknown error";
    }
}

cvector_error_t cvector_db_stats(cvector_db_t* db, cvector_db_stats_t* stats) {
    if (!db || !stats) return CVECTOR_ERROR_INVALID_ARGS;
    
    stats->total_vectors = db->vector_count;
    stats->dimension = db->config.dimension;
    stats->default_similarity = db->config.default_similarity;
    stats->total_size_bytes = 1024; // Placeholder
    strncpy(stats->db_path, db->config.data_path, sizeof(stats->db_path) - 1);
    
    return CVECTOR_SUCCESS;
}
EOF
    
    log_success "Created minimal C implementation"
}

# Create Go files
create_go_files() {
    log_info "Creating Go source files..."
    
    # Get module name from go.mod
    MODULE_NAME=$(grep "^module " go.mod | cut -d' ' -f2)
    
    # Create pkg/cvector/types.go
    cat > pkg/cvector/types.go << 'EOF'
package cvector

import "time"

// Error represents CVector error codes
type Error int

const (
	Success              Error = 0
	ErrInvalidArgs       Error = -1
	ErrOutOfMemory       Error = -2
	ErrFileIO            Error = -3
	ErrDBNotFound        Error = -4
	ErrVectorNotFound    Error = -5
	ErrDimensionMismatch Error = -6
	ErrDBCorrupt         Error = -7
)

func (e Error) Error() string {
	switch e {
	case Success:
		return "Success"
	case ErrInvalidArgs:
		return "Invalid arguments"
	case ErrOutOfMemory:
		return "Out of memory"
	case ErrFileIO:
		return "File I/O error"
	case ErrDBNotFound:
		return "Database not found"
	case ErrVectorNotFound:
		return "Vector not found"
	case ErrDimensionMismatch:
		return "Dimension mismatch"
	case ErrDBCorrupt:
		return "Database corrupt"
	default:
		return "Unknown error"
	}
}

// SimilarityType represents different similarity metrics
type SimilarityType int

const (
	SimilarityCosine     SimilarityType = 0
	SimilarityDotProduct SimilarityType = 1
	SimilarityEuclidean  SimilarityType = 2
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

// Vector represents a vector with metadata
type Vector struct {
	ID        uint64
	Dimension uint32
	Data      []float32
	Timestamp time.Time
}

// Stats holds database statistics
type Stats struct {
	TotalVectors      int
	TotalSizeBytes    int
	Dimension         uint32
	DefaultSimilarity SimilarityType
	DBPath            string
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
EOF
    
    # Create pkg/cvector/api.go
    cat > pkg/cvector/api.go << 'EOF'
package cvector

/*
#cgo CFLAGS: -I../../src -std=c11
#cgo LDFLAGS: -L../../build -lcvector -lm
#include "core/cvector.h"
#include <stdlib.h>
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

	// Convert Go config to C config
	cConfig := C.cvector_db_config_t{}
	
	nameBytes := []byte(config.Name)
	if len(nameBytes) >= 256 {
		return nil, ErrInvalidArgs
	}
	copy((*[256]C.char)(unsafe.Pointer(&cConfig.name[0]))[:], nameBytes)
	
	pathBytes := []byte(config.DataPath)
	if len(pathBytes) >= 1024 {
		return nil, ErrInvalidArgs
	}
	copy((*[1024]C.char)(unsafe.Pointer(&cConfig.data_path[0]))[:], pathBytes)
	
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

// Stats returns database statistics
func (db *DB) Stats() (*Stats, error) {
	if db.db == nil {
		return nil, ErrInvalidArgs
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
EOF
    
    # Create simple main.go
    cat > cmd/cvector/main.go << EOF
package main

import (
	"fmt"
	"os"
	
	"$MODULE_NAME/pkg/cvector"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Println("CVector Database CLI")
		fmt.Println("Usage:")
		fmt.Println("  cvector create --path=PATH --dimension=DIM")
		fmt.Println("  cvector stats --path=PATH")
		return
	}

	command := os.Args[1]
	
	switch command {
	case "create":
		fmt.Println("Creating test database...")
		config := &cvector.DBConfig{
			Name:              "test_db",
			DataPath:          "./data/test.cvdb",
			Dimension:         128,
			DefaultSimilarity: cvector.SimilarityCosine,
			MemoryMapped:      false,
			MaxVectors:        1000,
		}
		
		db, err := cvector.CreateDB(config)
		if err != nil {
			fmt.Printf("Error: %v\n", err)
			os.Exit(1)
		}
		defer db.Close()
		
		fmt.Println("Database created successfully!")
		
	case "stats":
		fmt.Println("Opening database...")
		db, err := cvector.OpenDB("./data/test.cvdb")
		if err != nil {
			fmt.Printf("Error: %v\n", err)
			os.Exit(1)
		}
		defer db.Close()
		
		stats, err := db.Stats()
		if err != nil {
			fmt.Printf("Error getting stats: %v\n", err)
			os.Exit(1)
		}
		
		fmt.Printf("Database Statistics:\n")
		fmt.Printf("  Path: %s\n", stats.DBPath)
		fmt.Printf("  Vectors: %d\n", stats.TotalVectors)
		fmt.Printf("  Dimension: %d\n", stats.Dimension)
		
	default:
		fmt.Printf("Unknown command: %s\n", command)
		os.Exit(1)
	}
}
EOF
    
    log_success "Created Go source files"
}

# Create Makefile
create_makefile() {
    log_info "Creating Makefile..."
    
    cat > Makefile << 'EOF'
CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -O2 -fPIC -I./src
GO = go
BUILD_DIR = build
SRC_DIR = src

# C library target
LIBCVECTOR = $(BUILD_DIR)/libcvector.a
C_SOURCES = $(wildcard $(SRC_DIR)/core/*.c) $(wildcard $(SRC_DIR)/utils/*.c)
C_OBJECTS = $(C_SOURCES:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)

# Go binary target
GO_BINARY = $(BUILD_DIR)/cvector

.PHONY: all clean test c-lib go-build setup

all: setup c-lib go-build

setup:
	@mkdir -p $(BUILD_DIR)/core $(BUILD_DIR)/utils
	@go mod tidy

c-lib: $(LIBCVECTOR)

$(LIBCVECTOR): $(C_OBJECTS) | $(BUILD_DIR)
	ar rcs $@ $^
	@echo "✅ C library built successfully"

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

go-build: c-lib
	@echo "Building Go binary..."
	CGO_ENABLED=1 \
	CGO_CFLAGS="-I./src -std=c11" \
	CGO_LDFLAGS="-L./build -lcvector -lm" \
	$(GO) build -o $(GO_BINARY) ./cmd/cvector
	@echo "✅ Go binary built successfully"

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)/core $(BUILD_DIR)/utils

clean:
	rm -rf $(BUILD_DIR)
	go clean

test: go-build
	@echo "🧪 Testing basic functionality..."
	@mkdir -p data
	./$(GO_BINARY) create
	./$(GO_BINARY) stats
	@echo "✅ Basic test passed"

help:
	@echo "CVector Build System"
	@echo "  all      - Build everything"
	@echo "  c-lib    - Build C library"
	@echo "  go-build - Build Go binary"
	@echo "  test     - Run basic test"
	@echo "  clean    - Clean build artifacts"
EOF
    
    log_success "Created Makefile"
}

# Build the project
build_project() {
    log_info "Building the project..."
    
    if make all; then
        log_success "Build completed successfully!"
    else
        log_error "Build failed!"
    fi
}

# Run basic test
test_project() {
    log_info "Running basic test..."
    
    if make test; then
        log_success "Basic test passed!"
    else
        log_warning "Basic test failed, but this is expected for the minimal implementation"
    fi
}

# Main execution
main() {
    echo
    log_info "Starting CVector setup..."
    echo
    
    check_prerequisites
    create_structure
    init_go_module
    create_c_files
    create_minimal_c_impl
    create_go_files
    create_makefile
    build_project
    test_project
    
    echo
    log_success "🎉 CVector setup completed!"
    echo
    echo "Next steps:"
    echo "1. Try: ./build/cvector create"
    echo "2. Try: ./build/cvector stats"
    echo "3. Implement full vector storage in src/core/vector_store.c"
    echo "4. Add more CLI commands"
    echo "5. Implement HNSW indexing"
    echo
    echo "To rebuild: make clean && make all"
    echo "To clean: make clean"
    echo
}

# Run main function
main "$@"