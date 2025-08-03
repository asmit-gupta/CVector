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

# Default target
all: setup c-lib go-build

# Setup - ensure directories exist and Go module is ready
setup:
	@mkdir -p $(BUILD_DIR)/core $(BUILD_DIR)/utils
	@go mod tidy

# Build C library FIRST
c-lib: $(LIBCVECTOR)

$(LIBCVECTOR): $(C_OBJECTS) | $(BUILD_DIR)
	ar rcs $@ $^
	@echo "C library built successfully"

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

# Build Go binary (depends on C library)
go-build: c-lib
	@echo "Building Go binary..."
	CGO_ENABLED=1 \
	CGO_CFLAGS="-I./src -std=c11" \
	CGO_LDFLAGS="-L./build -lcvector -lm" \
	$(GO) build -o $(GO_BINARY) ./cmd/cvector
	@echo "Go binary built successfully"

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)/core $(BUILD_DIR)/utils

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)
	go clean

# Run tests (build first)
test: test-c test-go

test-c: c-lib
	@echo "Running comprehensive tests..."
	$(CC) $(CFLAGS) -I./src -L./build -lcvector -lm tests/c/comprehensive_test.c -o $(BUILD_DIR)/comprehensive_test
	./$(BUILD_DIR)/comprehensive_test

test-go: go-build
	@echo "Running Go tests..."
	cd tests/go && CGO_ENABLED=1 CGO_CFLAGS="-I../../src" CGO_LDFLAGS="-L../../build -lcvector -lm" $(GO) test -v

# Install dependencies
deps:
	$(GO) mod tidy

# Development build (with debug symbols)
debug: CFLAGS += -g -DDEBUG
debug: all

# Help
help:
	@echo "CVector Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all      - Build everything (default)"
	@echo "  c-lib    - Build C library only"
	@echo "  go-build - Build Go binary only"
	@echo "  test     - Run all tests"
	@echo "  test-c   - Run C tests only"
	@echo "  test-go  - Run Go tests only"
	@echo "  clean    - Clean build artifacts"
	@echo "  setup    - Setup directories and Go module"
	@echo "  debug    - Build with debug symbols"
	@echo "  help     - Show this help"