package main

import (
	"flag"
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"strings"

	"github.com/asmit-gupta/cvector/pkg/cvector"
)

const (
	defaultDimension = 512
	defaultDBPath    = "./data/test.cvdb"
)

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	command := os.Args[1]
	args := os.Args[2:]

	switch command {
	case "create":
		handleCreate(args)
	case "insert":
		handleInsert(args)
	case "get":
		handleGet(args)
	case "delete":
		handleDelete(args)
	case "stats":
		handleStats(args)
	case "generate":
		handleGenerate(args)
	case "drop":
		handleDrop(args)
	case "search":
		handleSearch(args)
	default:
		fmt.Printf("Unknown command: %s\n", command)
		printUsage()
		os.Exit(1)
	}
}

func printUsage() {
	fmt.Println("CVector - Vector Database CLI")
	fmt.Println("")
	fmt.Println("Usage:")
	fmt.Println("  cvector create [--path=PATH] [--dimension=DIM] [--name=NAME]")
	fmt.Println("    Create a new vector database")
	fmt.Println("")
	fmt.Println("  cvector insert [--path=PATH] --id=ID --vector=\"1.0,2.0,3.0,...\"")
	fmt.Println("    Insert a vector into the database")
	fmt.Println("")
	fmt.Println("  cvector get [--path=PATH] --id=ID")
	fmt.Println("    Retrieve a vector by ID")
	fmt.Println("")
	fmt.Println("  cvector delete [--path=PATH] --id=ID")
	fmt.Println("    Delete a vector by ID")
	fmt.Println("")
	fmt.Println("  cvector stats [--path=PATH]")
	fmt.Println("    Show database statistics")
	fmt.Println("")
	fmt.Println("  cvector generate [--path=PATH] --count=N [--dimension=DIM]")
	fmt.Println("    Generate random test vectors")
	fmt.Println("")
	fmt.Println("  cvector drop --path=PATH")
	fmt.Println("    Drop (delete) a database")
	fmt.Println("")
	fmt.Println("  cvector search [--path=PATH] --vector=\"1.0,2.0,3.0,...\" [--top-k=K] [--similarity=TYPE]")
	fmt.Println("    Search for similar vectors")
	fmt.Println("")
	fmt.Println("Options:")
	fmt.Printf("  --path        Database file path (default: %s)\n", defaultDBPath)
	fmt.Printf("  --dimension   Vector dimension (default: %d)\n", defaultDimension)
	fmt.Println("  --name        Database name")
	fmt.Println("  --id          Vector ID")
	fmt.Println("  --vector      Vector data as comma-separated floats")
	fmt.Println("  --count       Number of vectors to generate")
	fmt.Println("  --top-k       Number of results to return (default: 10)")
	fmt.Println("  --similarity  Similarity type: cosine, dot, euclidean (default: cosine)")
}

func handleCreate(args []string) {
	fs := flag.NewFlagSet("create", flag.ExitOnError)
	path := fs.String("path", defaultDBPath, "Database path")
	dimension := fs.Int("dimension", defaultDimension, "Vector dimension")
	name := fs.String("name", "test_db", "Database name")

	fs.Parse(args)

	config := &cvector.DBConfig{
		Name:              *name,
		DataPath:          *path,
		Dimension:         uint32(*dimension),
		DefaultSimilarity: cvector.SimilarityCosine,
		MemoryMapped:      false,
		MaxVectors:        1000000,
	}

	fmt.Printf("Creating database: %s\n", *path)
	fmt.Printf("  Name: %s\n", *name)
	fmt.Printf("  Dimension: %d\n", *dimension)

	db, err := cvector.CreateDB(config)
	if err != nil {
		fmt.Printf("Error creating database: %v\n", err)
		os.Exit(1)
	}
	defer db.Close()

	fmt.Printf("Database created successfully!\n")
}

func handleInsert(args []string) {
	fs := flag.NewFlagSet("insert", flag.ExitOnError)
	path := fs.String("path", defaultDBPath, "Database path")
	id := fs.Uint64("id", 0, "Vector ID")
	vectorStr := fs.String("vector", "", "Vector data (comma-separated floats)")

	fs.Parse(args)

	if *id == 0 {
		fmt.Println("Error: --id is required")
		os.Exit(1)
	}

	if *vectorStr == "" {
		fmt.Println("Error: --vector is required")
		os.Exit(1)
	}

	// Parse vector data
	vectorData, err := parseVectorString(*vectorStr)
	if err != nil {
		fmt.Printf("Error parsing vector: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Opening database: %s\n", *path)
	db, err := cvector.OpenDB(*path)
	if err != nil {
		fmt.Printf("Error opening database: %v\n", err)
		os.Exit(1)
	}
	defer db.Close()

	vector := cvector.NewVector(*id, vectorData)
	fmt.Printf("Inserting vector ID %d (dimension: %d)\n", *id, len(vectorData))

	err = db.Insert(vector)
	if err != nil {
		fmt.Printf("Error inserting vector: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Vector inserted successfully!\n")
}

func handleGet(args []string) {
	fs := flag.NewFlagSet("get", flag.ExitOnError)
	path := fs.String("path", defaultDBPath, "Database path")
	id := fs.Uint64("id", 0, "Vector ID")

	fs.Parse(args)

	if *id == 0 {
		fmt.Println("Error: --id is required")
		os.Exit(1)
	}

	fmt.Printf("Opening database: %s\n", *path)
	db, err := cvector.OpenDB(*path)
	if err != nil {
		fmt.Printf("Error opening database: %v\n", err)
		os.Exit(1)
	}
	defer db.Close()

	fmt.Printf("Retrieving vector ID %d\n", *id)
	vector, err := db.Get(*id)
	if err != nil {
		fmt.Printf("Error retrieving vector: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Vector found:\n")
	fmt.Printf("  ID: %d\n", vector.ID)
	fmt.Printf("  Dimension: %d\n", vector.Dimension)
	fmt.Printf("  Timestamp: %s\n", vector.Timestamp.Format("2006-01-02 15:04:05"))
	fmt.Printf("  Data: [%s]\n", formatVector(vector.Data))
}

func handleDelete(args []string) {
	fs := flag.NewFlagSet("delete", flag.ExitOnError)
	path := fs.String("path", defaultDBPath, "Database path")
	id := fs.Uint64("id", 0, "Vector ID")

	fs.Parse(args)

	if *id == 0 {
		fmt.Println("Error: --id is required")
		os.Exit(1)
	}

	fmt.Printf("Opening database: %s\n", *path)
	db, err := cvector.OpenDB(*path)
	if err != nil {
		fmt.Printf("Error opening database: %v\n", err)
		os.Exit(1)
	}
	defer db.Close()

	fmt.Printf("Deleting vector ID %d\n", *id)
	err = db.Delete(*id)
	if err != nil {
		fmt.Printf("Error deleting vector: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Vector deleted successfully!\n")
}

func handleStats(args []string) {
	fs := flag.NewFlagSet("stats", flag.ExitOnError)
	path := fs.String("path", defaultDBPath, "Database path")

	fs.Parse(args)

	fmt.Printf("Opening database: %s\n", *path)
	db, err := cvector.OpenDB(*path)
	if err != nil {
		fmt.Printf("Error opening database: %v\n", err)
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
	fmt.Printf("  Total Vectors: %d\n", stats.TotalVectors)
	fmt.Printf("  Dimension: %d\n", stats.Dimension)
	fmt.Printf("  File Size: %d bytes (%.2f MB)\n", stats.TotalSizeBytes, float64(stats.TotalSizeBytes)/(1024*1024))
	fmt.Printf("  Default Similarity: %v\n", stats.DefaultSimilarity)
}

func handleGenerate(args []string) {
	fs := flag.NewFlagSet("generate", flag.ExitOnError)
	path := fs.String("path", defaultDBPath, "Database path")
	count := fs.Int("count", 0, "Number of vectors to generate")
	dimension := fs.Int("dimension", defaultDimension, "Vector dimension")

	fs.Parse(args)

	if *count <= 0 {
		fmt.Println("Error: --count must be greater than 0")
		os.Exit(1)
	}

	fmt.Printf("Opening database: %s\n", *path)
	db, err := cvector.OpenDB(*path)
	if err != nil {
		fmt.Printf("Error opening database: %v\n", err)
		os.Exit(1)
	}
	defer db.Close()

	fmt.Printf("Generating %d random vectors (dimension: %d)\n", *count, *dimension)

	for i := 0; i < *count; i++ {
		// Generate random vector
		data := make([]float32, *dimension)
		for j := range data {
			data[j] = rand.Float32()*2 - 1 // Random float between -1 and 1
		}

		vector := cvector.NewVector(uint64(i+1), data)
		err = db.Insert(vector)
		if err != nil {
			fmt.Printf("Error inserting vector %d: %v\n", i+1, err)
			continue
		}

		if (i+1)%100 == 0 {
			fmt.Printf("  Generated %d vectors...\n", i+1)
		}
	}

	fmt.Printf("Generated %d vectors successfully!\n", *count)
}

func handleDrop(args []string) {
	fs := flag.NewFlagSet("drop", flag.ExitOnError)
	path := fs.String("path", "", "Database path")

	fs.Parse(args)

	if *path == "" {
		fmt.Println("Error: --path is required for drop command")
		os.Exit(1)
	}

	fmt.Printf("Dropping database: %s\n", *path)
	fmt.Print("Are you sure? (y/N): ")
	var response string
	fmt.Scanln(&response)

	if strings.ToLower(response) != "y" && strings.ToLower(response) != "yes" {
		fmt.Println("Operation cancelled.")
		return
	}

	err := cvector.DropDB(*path)
	if err != nil {
		fmt.Printf("Error dropping database: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Database dropped successfully!\n")
}

func handleSearch(args []string) {
	fs := flag.NewFlagSet("search", flag.ExitOnError)
	path := fs.String("path", defaultDBPath, "Database path")
	vectorStr := fs.String("vector", "", "Query vector data (comma-separated floats)")
	topK := fs.Int("top-k", 10, "Number of results to return")
	similarityStr := fs.String("similarity", "cosine", "Similarity type (cosine, dot, euclidean)")

	fs.Parse(args)

	if *vectorStr == "" {
		fmt.Println("Error: --vector is required")
		os.Exit(1)
	}

	if *topK <= 0 {
		fmt.Println("Error: --top-k must be greater than 0")
		os.Exit(1)
	}

	// Parse vector data
	queryVector, err := parseVectorString(*vectorStr)
	if err != nil {
		fmt.Printf("Error parsing query vector: %v\n", err)
		os.Exit(1)
	}

	// Parse similarity type
	var similarity cvector.SimilarityType
	switch strings.ToLower(*similarityStr) {
	case "cosine":
		similarity = cvector.SimilarityCosine
	case "dot", "dotproduct":
		similarity = cvector.SimilarityDotProduct
	case "euclidean", "l2":
		similarity = cvector.SimilarityEuclidean
	default:
		fmt.Printf("Error: unknown similarity type '%s'. Use cosine, dot, or euclidean\n", *similarityStr)
		os.Exit(1)
	}

	fmt.Printf("Opening database: %s\n", *path)
	db, err := cvector.OpenDB(*path)
	if err != nil {
		fmt.Printf("Error opening database: %v\n", err)
		os.Exit(1)
	}
	defer db.Close()

	query := &cvector.Query{
		QueryVector:   queryVector,
		TopK:          uint32(*topK),
		Similarity:    similarity,
		MinSimilarity: 0.0, // No minimum similarity filter
	}

	fmt.Printf("Searching for similar vectors (top-%d, similarity: %s, dimension: %d)\n", 
		*topK, *similarityStr, len(queryVector))

	results, err := db.Search(query)
	if err != nil {
		fmt.Printf("Error searching: %v\n", err)
		os.Exit(1)
	}

	if len(results) == 0 {
		fmt.Println("No similar vectors found.")
		return
	}

	fmt.Printf("\nSearch Results (%d found):\n", len(results))
	fmt.Println("Rank | Vector ID | Similarity Score")
	fmt.Println("-----|-----------|----------------")
	for i, result := range results {
		fmt.Printf("%-4d | %-9d | %.6f\n", i+1, result.ID, result.Similarity)
	}
}

// Helper functions

func parseVectorString(vectorStr string) ([]float32, error) {
	parts := strings.Split(vectorStr, ",")
	data := make([]float32, len(parts))

	for i, part := range parts {
		val, err := strconv.ParseFloat(strings.TrimSpace(part), 32)
		if err != nil {
			return nil, fmt.Errorf("invalid float value: %s", part)
		}
		data[i] = float32(val)
	}

	return data, nil
}

func formatVector(data []float32) string {
	if len(data) <= 10 {
		strs := make([]string, len(data))
		for i, v := range data {
			strs[i] = fmt.Sprintf("%.3f", v)
		}
		return strings.Join(strs, ", ")
	}

	// Show first 5 and last 5 values for long vectors
	var parts []string
	for i := 0; i < 5; i++ {
		parts = append(parts, fmt.Sprintf("%.3f", data[i]))
	}
	parts = append(parts, "...")
	for i := len(data) - 5; i < len(data); i++ {
		parts = append(parts, fmt.Sprintf("%.3f", data[i]))
	}
	return strings.Join(parts, ", ")
}