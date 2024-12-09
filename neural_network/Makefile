# Compiler and flags
NVCC = nvcc
CXXFLAGS = -std=c++11 -O3

# Directories
SRC_DIR = src
INC_DIR = include
BIN_DIR = bin
OBJ_DIR = obj

# Output binary
TARGET = $(BIN_DIR)/neural_network

# Source and object files
SRC_FILES = $(wildcard $(SRC_DIR)/*.cu)
OBJ_FILES = $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(SRC_FILES))

# Rules
all: setup $(TARGET)

setup:
	@mkdir -p $(BIN_DIR) $(OBJ_DIR)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(CXXFLAGS) -I$(INC_DIR) -c $< -o $@

$(TARGET): $(OBJ_FILES)
	$(NVCC) $(CXXFLAGS) $(OBJ_FILES) -o $@

clean:
	rm -rf $(BIN_DIR) $(OBJ_DIR)

run: all
	./$(TARGET)

.PHONY: all setup clean run
