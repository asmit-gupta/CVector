package main

import (
	"fmt"
	"math/rand"
	"os"
	"runtime"
	"strings"
	"time"

	"github.com/asmit-gupta/cvector/pkg/cvector"
)

const (
	testDBPath    = "./data/bench.cvdb"
	testDimension = 512
)

type BenchmarkResult struct {
	OperationType string
	Duration      time.Duration
	Operations    int
	QPS           float64
	AvgLatency    time.Duration
	MinLatency    time.Duration
	MaxLatency    time.Duration
	MemoryUsed    uint64
	ErrorCount    int
}

type PerformanceStats struct {
	Results []BenchmarkResult
	TotalTime time.Duration
	DBStats   *cvector.Stats
}

func main() {
	fmt.Println("🚀 CVector Performance Benchmark Suite")
	fmt.Println("=====================================")
	
	// Clean up any existing test database
	cleanup()
	defer cleanup()
	
	// Run comprehensive benchmarks
	stats := runFullBenchmark()
	
	// Display results
	displayResults(stats)
	
	// Save results to file
	saveResults(stats)
}

func cleanup() {
	os.Remove(testDBPath)
}

func runFullBenchmark() *PerformanceStats {
	stats := &PerformanceStats{
		Results: make([]BenchmarkResult, 0),
	}
	
	startTime := time.Now()
	
	// Create database
	db := createTestDB()
	defer db.Close()
	
	// Run different benchmark scenarios
	fmt.Println("\n📊 Running Benchmarks...")
	
	// 1. Insert Performance
	stats.Results = append(stats.Results, benchmarkInsert(db, 1000, "Bulk Insert (1K vectors)"))
	stats.Results = append(stats.Results, benchmarkInsert(db, 100, "Single Insert (100 vectors)"))
	
	// 2. Query Performance  
	stats.Results = append(stats.Results, benchmarkGet(db, 500, "Random Get (500 queries)"))
	stats.Results = append(stats.Results, benchmarkGet(db, 1000, "Sequential Get (1K queries)"))
	
	// 3. Mixed Workload
	stats.Results = append(stats.Results, benchmarkMixedWorkload(db, 500))
	
	// 4. Delete Performance
	stats.Results = append(stats.Results, benchmarkDelete(db, 200, "Bulk Delete (200 vectors)"))
	
	// 5. Large Batch Insert
	stats.Results = append(stats.Results, benchmarkInsert(db, 5000, "Large Batch Insert (5K vectors)"))
	
	stats.TotalTime = time.Since(startTime)
	
	// Get final database stats
	dbStats, _ := db.Stats()
	stats.DBStats = dbStats
	
	return stats
}

func createTestDB() *cvector.DB {
	config := &cvector.DBConfig{
		Name:              "benchmark_db",
		DataPath:          testDBPath,
		Dimension:         testDimension,
		DefaultSimilarity: cvector.SimilarityCosine,
		MemoryMapped:      false,
		MaxVectors:        100000,
	}
	
	db, err := cvector.CreateDB(config)
	if err != nil {
		panic(fmt.Sprintf("Failed to create test database: %v", err))
	}
	
	return db
}

func benchmarkInsert(db *cvector.DB, count int, description string) BenchmarkResult {
	fmt.Printf("  🔄 %s...", description)
	
	result := BenchmarkResult{
		OperationType: description,
		Operations:    count,
		MinLatency:    time.Hour, // Will be updated
	}
	
	var memBefore, memAfter runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&memBefore)
	
	latencies := make([]time.Duration, count)
	errors := 0
	
	// Get current stats to start IDs from a safe number
	stats, _ := db.Stats()
	startID := uint64(stats.TotalVectors + 1000) // Start from a safe ID
	
	startTime := time.Now()
	
	for i := 0; i < count; i++ {
		// Generate random vector
		data := generateRandomVector(testDimension)
		vector := cvector.NewVector(startID+uint64(i), data) // Use unique IDs
		
		opStart := time.Now()
		err := db.Insert(vector)
		latency := time.Since(opStart)
		
		latencies[i] = latency
		
		if err != nil {
			errors++
			// Log first few errors for debugging
			if errors <= 3 {
				fmt.Printf("\n    Error inserting vector %d: %v", startID+uint64(i), err)
			}
		}
		
		// Update min/max latency
		if latency < result.MinLatency {
			result.MinLatency = latency
		}
		if latency > result.MaxLatency {
			result.MaxLatency = latency
		}
	}
	
	result.Duration = time.Since(startTime)
	result.QPS = float64(count) / result.Duration.Seconds()
	result.AvgLatency = result.Duration / time.Duration(count)
	result.ErrorCount = errors
	
	runtime.ReadMemStats(&memAfter)
	result.MemoryUsed = memAfter.Alloc - memBefore.Alloc
	
	fmt.Printf(" ✅ %.2f QPS", result.QPS)
	if errors > 0 {
		fmt.Printf(" (⚠️ %d errors)", errors)
	}
	fmt.Println()
	return result
}

func benchmarkGet(db *cvector.DB, count int, description string) BenchmarkResult {
	fmt.Printf("  🔍 %s...", description)
	
	result := BenchmarkResult{
		OperationType: description,
		Operations:    count,
		MinLatency:    time.Hour,
	}
	
	var memBefore, memAfter runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&memBefore)
	
	errors := 0
	startTime := time.Now()
	
	for i := 0; i < count; i++ {
		// Try to get existing vector (may not exist, that's ok)
		id := uint64(rand.Intn(1000) + 1)
		
		opStart := time.Now()
		_, err := db.Get(id)
		latency := time.Since(opStart)
		
		if err != nil && err != cvector.ErrVectorNotFound {
			errors++
		}
		
		// Update min/max latency
		if latency < result.MinLatency {
			result.MinLatency = latency
		}
		if latency > result.MaxLatency {
			result.MaxLatency = latency
		}
	}
	
	result.Duration = time.Since(startTime)
	result.QPS = float64(count) / result.Duration.Seconds()
	result.AvgLatency = result.Duration / time.Duration(count)
	result.ErrorCount = errors
	
	runtime.ReadMemStats(&memAfter)
	result.MemoryUsed = memAfter.Alloc - memBefore.Alloc
	
	fmt.Printf(" ✅ %.2f QPS\n", result.QPS)
	return result
}

func benchmarkDelete(db *cvector.DB, count int, description string) BenchmarkResult {
	fmt.Printf("  🗑️  %s...", description)
	
	result := BenchmarkResult{
		OperationType: description,
		Operations:    count,
		MinLatency:    time.Hour,
	}
	
	errors := 0
	startTime := time.Now()
	
	for i := 0; i < count; i++ {
		id := uint64(i + 1)
		
		opStart := time.Now()
		err := db.Delete(id)
		latency := time.Since(opStart)
		
		if err != nil {
			errors++
		}
		
		// Update min/max latency
		if latency < result.MinLatency {
			result.MinLatency = latency
		}
		if latency > result.MaxLatency {
			result.MaxLatency = latency
		}
	}
	
	result.Duration = time.Since(startTime)
	result.QPS = float64(count) / result.Duration.Seconds()
	result.AvgLatency = result.Duration / time.Duration(count)
	result.ErrorCount = errors
	
	fmt.Printf(" ✅ %.2f QPS\n", result.QPS)
	return result
}

func benchmarkMixedWorkload(db *cvector.DB, operations int) BenchmarkResult {
	fmt.Printf("  🔀 Mixed Workload (%d ops)...", operations)
	
	result := BenchmarkResult{
		OperationType: "Mixed Workload (70% read, 20% write, 10% delete)",
		Operations:    operations,
		MinLatency:    time.Hour,
	}
	
	errors := 0
	
	// Get current stats for safe ID ranges
	stats, _ := db.Stats()
	writeStartID := uint64(stats.TotalVectors + 10000) // Safe range for new inserts
	
	startTime := time.Now()
	
	for i := 0; i < operations; i++ {
		opType := rand.Float64()
		
		opStart := time.Now()
		var err error
		
		switch {
		case opType < 0.7: // 70% reads
			readID := uint64(rand.Intn(int(stats.TotalVectors)) + 1)
			_, err = db.Get(readID)
			if err == cvector.ErrVectorNotFound {
				err = nil // Expected error
			}
		case opType < 0.9: // 20% writes
			data := generateRandomVector(testDimension)
			vector := cvector.NewVector(writeStartID+uint64(i), data)
			err = db.Insert(vector)
		default: // 10% deletes
			deleteID := uint64(rand.Intn(int(stats.TotalVectors)) + 1)
			err = db.Delete(deleteID)
		}
		
		latency := time.Since(opStart)
		
		if err != nil {
			errors++
		}
		
		// Update min/max latency
		if latency < result.MinLatency {
			result.MinLatency = latency
		}
		if latency > result.MaxLatency {
			result.MaxLatency = latency
		}
	}
	
	result.Duration = time.Since(startTime)
	result.QPS = float64(operations) / result.Duration.Seconds()
	result.AvgLatency = result.Duration / time.Duration(operations)
	result.ErrorCount = errors
	
	fmt.Printf(" ✅ %.2f QPS", result.QPS)
	if errors > 0 {
		fmt.Printf(" (⚠️ %d errors)", errors)
	}
	fmt.Println()
	return result
}

func generateRandomVector(dimension int) []float32 {
	data := make([]float32, dimension)
	for i := range data {
		data[i] = rand.Float32()*2 - 1 // Random float between -1 and 1
	}
	return data
}

func displayResults(stats *PerformanceStats) {
	fmt.Println("\n📈 Performance Results")
	fmt.Println("=====================")
	
	fmt.Printf("Total Benchmark Time: %v\n\n", stats.TotalTime)
	
	// Display table header
	fmt.Printf("%-35s %-12s %-10s %-12s %-12s %-12s %-10s\n",
		"Operation", "Duration", "Ops", "QPS", "Avg Latency", "Max Latency", "Errors")
	fmt.Println(strings.Repeat("-", 105))
	
	for _, result := range stats.Results {
		fmt.Printf("%-35s %-12v %-10d %-12.2f %-12v %-12v %-10d\n",
			result.OperationType,
			result.Duration.Round(time.Millisecond),
			result.Operations,
			result.QPS,
			result.AvgLatency.Round(time.Microsecond),
			result.MaxLatency.Round(time.Microsecond),
			result.ErrorCount)
	}
	
	if stats.DBStats != nil {
		fmt.Println("\n💾 Database Statistics")
		fmt.Println("=====================")
		fmt.Printf("Total Vectors: %d\n", stats.DBStats.TotalVectors)
		fmt.Printf("Dimension: %d\n", stats.DBStats.Dimension)
		fmt.Printf("File Size: %d bytes (%.2f MB)\n", 
			stats.DBStats.TotalSizeBytes, 
			float64(stats.DBStats.TotalSizeBytes)/(1024*1024))
		fmt.Printf("Default Similarity: %v\n", stats.DBStats.DefaultSimilarity)
		fmt.Printf("Database Path: %s\n", stats.DBStats.DBPath)
	}
	
	fmt.Println("\n🎯 Summary")
	fmt.Println("==========")
	
	totalOps := 0
	totalQPS := 0.0
	for _, result := range stats.Results {
		totalOps += result.Operations
		totalQPS += result.QPS
	}
	
	avgQPS := totalQPS / float64(len(stats.Results))
	
	fmt.Printf("Total Operations: %d\n", totalOps)
	fmt.Printf("Average QPS: %.2f\n", avgQPS)
	fmt.Printf("Operations per Second: %.2f\n", float64(totalOps)/stats.TotalTime.Seconds())
	
	// System info
	fmt.Println("\n💻 System Information")
	fmt.Println("====================")
	fmt.Printf("Go Version: %s\n", runtime.Version())
	fmt.Printf("OS/Arch: %s/%s\n", runtime.GOOS, runtime.GOARCH)
	fmt.Printf("CPUs: %d\n", runtime.NumCPU())
	
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	fmt.Printf("Memory Usage: %.2f MB\n", float64(m.Alloc)/(1024*1024))
}

func saveResults(stats *PerformanceStats) {
	filename := fmt.Sprintf("benchmark_results_%s.txt", time.Now().Format("20060102_150405"))
	
	file, err := os.Create(filename)
	if err != nil {
		fmt.Printf("Warning: Could not save results to file: %v\n", err)
		return
	}
	defer file.Close()
	
	// Write results to file (similar format as display)
	fmt.Fprintf(file, "CVector Performance Benchmark Results\n")
	fmt.Fprintf(file, "Generated: %s\n", time.Now().Format("2006-01-02 15:04:05"))
	fmt.Fprintf(file, "Total Time: %v\n\n", stats.TotalTime)
	
	for _, result := range stats.Results {
		fmt.Fprintf(file, "%s: %.2f QPS, %v avg latency, %d errors\n",
			result.OperationType, result.QPS, result.AvgLatency, result.ErrorCount)
	}
	
	fmt.Printf("\n💾 Results saved to: %s\n", filename)
}