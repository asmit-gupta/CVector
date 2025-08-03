package main

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/asmit-gupta/cvector/pkg/cvector"
)

const (
	testDBPath    = "./test_go.cvdb"
	testDimension = 128
)

func cleanupTestDB(t *testing.T) {
	os.Remove(testDBPath)
	// Also remove directory if empty
	dir := filepath.Dir(testDBPath)
	os.Remove(dir)
}

func createTestDB(t *testing.T) *cvector.DB {
	// Ensure directory exists
	dir := filepath.Dir(testDBPath)
	os.MkdirAll(dir, 0755)

	config := &cvector.DBConfig{
		Name:              "test_db",
		DataPath:          testDBPath,
		Dimension:         testDimension,
		DefaultSimilarity: cvector.SimilarityCosine,
		MemoryMapped:      false,
		MaxVectors:        1000,
	}

	db, err := cvector.CreateDB(config)
	if err != nil {
		t.Fatalf("Failed to create test database: %v", err)
	}

	return db
}

func createTestVector(id uint64, dimension int) *cvector.Vector {
	data := make([]float32, dimension)
	for i := range data {
		data[i] = float32(id*uint64(dimension)+uint64(i)) / 1000.0
	}
	return cvector.NewVector(id, data)
}

func TestDatabaseCreate(t *testing.T) {
	cleanupTestDB(t)
	defer cleanupTestDB(t)

	db := createTestDB(t)
	defer db.Close()

	// Verify database was created correctly
	stats, err := db.Stats()
	if err != nil {
		t.Fatalf("Failed to get stats: %v", err)
	}

	if stats.Dimension != testDimension {
		t.Errorf("Expected dimension %d, got %d", testDimension, stats.Dimension)
	}

	if stats.TotalVectors != 0 {
		t.Errorf("Expected 0 vectors, got %d", stats.TotalVectors)
	}

	if stats.DefaultSimilarity != cvector.SimilarityCosine {
		t.Errorf("Expected cosine similarity, got %v", stats.DefaultSimilarity)
	}
}

func TestDatabaseOpen(t *testing.T) {
	cleanupTestDB(t)
	defer cleanupTestDB(t)

	// Create database
	db := createTestDB(t)
	db.Close()

	// Reopen database
	db, err := cvector.OpenDB(testDBPath)
	if err != nil {
		t.Fatalf("Failed to open database: %v", err)
	}
	defer db.Close()

	// Verify it opened correctly
	stats, err := db.Stats()
	if err != nil {
		t.Fatalf("Failed to get stats: %v", err)
	}

	if stats.Dimension != testDimension {
		t.Errorf("Expected dimension %d, got %d", testDimension, stats.Dimension)
	}
}

func TestVectorInsert(t *testing.T) {
	cleanupTestDB(t)
	defer cleanupTestDB(t)

	db := createTestDB(t)
	defer db.Close()

	vector := createTestVector(1, testDimension)
	err := db.Insert(vector)
	if err != nil {
		t.Fatalf("Failed to insert vector: %v", err)
	}

	// Verify stats updated
	stats, err := db.Stats()
	if err != nil {
		t.Fatalf("Failed to get stats: %v", err)
	}

	if stats.TotalVectors != 1 {
		t.Errorf("Expected 1 vector, got %d", stats.TotalVectors)
	}
}

func TestVectorGet(t *testing.T) {
	cleanupTestDB(t)
	defer cleanupTestDB(t)

	db := createTestDB(t)
	defer db.Close()

	// Insert test vector
	original := createTestVector(42, testDimension)
	err := db.Insert(original)
	if err != nil {
		t.Fatalf("Failed to insert vector: %v", err)
	}

	// Retrieve vector
	retrieved, err := db.Get(42)
	if err != nil {
		t.Fatalf("Failed to get vector: %v", err)
	}

	// Verify data matches
	if retrieved.ID != original.ID {
		t.Errorf("Expected ID %d, got %d", original.ID, retrieved.ID)
	}

	if retrieved.Dimension != original.Dimension {
		t.Errorf("Expected dimension %d, got %d", original.Dimension, retrieved.Dimension)
	}

	if len(retrieved.Data) != len(original.Data) {
		t.Errorf("Expected data length %d, got %d", len(original.Data), len(retrieved.Data))
	}

	for i, v := range original.Data {
		if retrieved.Data[i] != v {
			t.Errorf("Data mismatch at index %d: expected %f, got %f", i, v, retrieved.Data[i])
		}
	}
}

func TestVectorDelete(t *testing.T) {
	cleanupTestDB(t)
	defer cleanupTestDB(t)

	db := createTestDB(t)
	defer db.Close()

	// Insert test vector
	vector := createTestVector(100, testDimension)
	err := db.Insert(vector)
	if err != nil {
		t.Fatalf("Failed to insert vector: %v", err)
	}

	// Delete vector
	err = db.Delete(100)
	if err != nil {
		t.Fatalf("Failed to delete vector: %v", err)
	}

	// Verify it's gone
	_, err = db.Get(100)
	if err == nil {
		t.Error("Expected error when getting deleted vector")
	}

	if err != cvector.ErrVectorNotFound {
		t.Errorf("Expected ErrVectorNotFound, got %v", err)
	}
}

func TestMultipleVectors(t *testing.T) {
	cleanupTestDB(t)
	defer cleanupTestDB(t)

	db := createTestDB(t)
	defer db.Close()

	const numVectors = 10

	// Insert multiple vectors
	for i := 1; i <= numVectors; i++ {
		vector := createTestVector(uint64(i), testDimension)
		err := db.Insert(vector)
		if err != nil {
			t.Fatalf("Failed to insert vector %d: %v", i, err)
		}
	}

	// Verify all vectors can be retrieved
	for i := 1; i <= numVectors; i++ {
		retrieved, err := db.Get(uint64(i))
		if err != nil {
			t.Fatalf("Failed to get vector %d: %v", i, err)
		}

		if retrieved.ID != uint64(i) {
			t.Errorf("Expected ID %d, got %d", i, retrieved.ID)
		}
	}

	// Check stats
	stats, err := db.Stats()
	if err != nil {
		t.Fatalf("Failed to get stats: %v", err)
	}

	if stats.TotalVectors != numVectors {
		t.Errorf("Expected %d vectors, got %d", numVectors, stats.TotalVectors)
	}
}

func TestErrorConditions(t *testing.T) {
	cleanupTestDB(t)
	defer cleanupTestDB(t)

	// Test opening non-existent database
	_, err := cvector.OpenDB("non_existent.cvdb")
	if err == nil {
		t.Error("Expected error when opening non-existent database")
	}

	if err != cvector.ErrDBNotFound {
		t.Errorf("Expected ErrDBNotFound, got %v", err)
	}

	// Test invalid config
	config := &cvector.DBConfig{
		Name:              "test",
		DataPath:          testDBPath,
		Dimension:         0, // Invalid
		DefaultSimilarity: cvector.SimilarityCosine,
	}

	_, err = cvector.CreateDB(config)
	if err == nil {
		t.Error("Expected error with invalid dimension")
	}

	// Test inserting vector with wrong dimension
	db := createTestDB(t)
	defer db.Close()

	wrongDimVector := createTestVector(1, testDimension+1)
	err = db.Insert(wrongDimVector)
	if err == nil {
		t.Error("Expected error when inserting vector with wrong dimension")
	}

	if err != cvector.ErrDimensionMismatch {
		t.Errorf("Expected ErrDimensionMismatch, got %v", err)
	}

	// Test getting non-existent vector
	_, err = db.Get(999)
	if err == nil {
		t.Error("Expected error when getting non-existent vector")
	}

	if err != cvector.ErrVectorNotFound {
		t.Errorf("Expected ErrVectorNotFound, got %v", err)
	}
}

func TestVectorCreation(t *testing.T) {
	data := []float32{1.0, 2.0, 3.0, 4.0}
	vector := cvector.NewVector(123, data)

	if vector.ID != 123 {
		t.Errorf("Expected ID 123, got %d", vector.ID)
	}

	if vector.Dimension != 4 {
		t.Errorf("Expected dimension 4, got %d", vector.Dimension)
	}

	if len(vector.Data) != 4 {
		t.Errorf("Expected data length 4, got %d", len(vector.Data))
	}

	for i, v := range data {
		if vector.Data[i] != v {
			t.Errorf("Data mismatch at index %d: expected %f, got %f", i, v, vector.Data[i])
		}
	}

	// Check timestamp is recent
	if time.Since(vector.Timestamp) > time.Second {
		t.Error("Vector timestamp is not recent")
	}
}

func TestDatabaseDrop(t *testing.T) {
	cleanupTestDB(t)

	// Create database
	db := createTestDB(t)
	db.Close()

	// Verify file exists
	if _, err := os.Stat(testDBPath); os.IsNotExist(err) {
		t.Fatal("Database file should exist")
	}

	// Drop database
	err := cvector.DropDB(testDBPath)
	if err != nil {
		t.Fatalf("Failed to drop database: %v", err)
	}

	// Verify file is gone
	if _, err := os.Stat(testDBPath); !os.IsNotExist(err) {
		t.Error("Database file should be deleted")
	}
}

func BenchmarkVectorInsert(b *testing.B) {
	cleanupTestDB(nil)
	defer cleanupTestDB(nil)

	// Ensure directory exists
	dir := filepath.Dir(testDBPath)
	os.MkdirAll(dir, 0755)

	config := &cvector.DBConfig{
		Name:              "bench_db",
		DataPath:          testDBPath,
		Dimension:         testDimension,
		DefaultSimilarity: cvector.SimilarityCosine,
		MemoryMapped:      false,
		MaxVectors:        b.N,
	}

	db, err := cvector.CreateDB(config)
	if err != nil {
		b.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		vector := createTestVector(uint64(i+1), testDimension)
		err := db.Insert(vector)
		if err != nil {
			b.Fatalf("Failed to insert vector %d: %v", i+1, err)
		}
	}
}

func BenchmarkVectorGet(b *testing.B) {
	cleanupTestDB(nil)
	defer cleanupTestDB(nil)

	// Setup database with vectors
	dir := filepath.Dir(testDBPath)
	os.MkdirAll(dir, 0755)

	config := &cvector.DBConfig{
		Name:              "bench_db",
		DataPath:          testDBPath,
		Dimension:         testDimension,
		DefaultSimilarity: cvector.SimilarityCosine,
		MemoryMapped:      false,
		MaxVectors:        b.N,
	}

	db, err := cvector.CreateDB(config)
	if err != nil {
		b.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	// Insert test vectors
	for i := 0; i < b.N; i++ {
		vector := createTestVector(uint64(i+1), testDimension)
		err := db.Insert(vector)
		if err != nil {
			b.Fatalf("Failed to insert vector %d: %v", i+1, err)
		}
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, err := db.Get(uint64(i + 1))
		if err != nil {
			b.Fatalf("Failed to get vector %d: %v", i+1, err)
		}
	}
}