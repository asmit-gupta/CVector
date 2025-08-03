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