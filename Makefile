CU_LIBS_DIR := $(CURDIR)/cu_libs
CU_SOURCES := $(CURDIR)/tests/test_kernels.cu
CU_OBJECTS := $(CU_LIBS_DIR)/test_kernels.cu.o

.PHONY: all run-test clean

all: run-test

$(CU_LIBS_DIR)/%.cu.o: $(CURDIR)/tests/%.cu
	@mkdir -p $(CU_LIBS_DIR) > /dev/null 2>&1 || true
	nvcc -arch=sm_90 -std=c++14 -I $(CURDIR) -c $< -o $@


$(CURDIR)/output/esie-test: $(CU_OBJECTS)
	@mkdir -p $(CURDIR)/output > /dev/null 2>&1 || true
	g++ -std=c++14 -I $(CURDIR) $(CURDIR)/esie/models/*.cpp $(CURDIR)/tests/main.cpp $(CU_OBJECTS) -L /usr/local/cuda/lib64 -lcudart -o $(CURDIR)/output/esie-test

run-test: $(CURDIR)/output/esie-test
	${CURDIR}/output/esie-test
	@echo "Run completed"

clean:
	rm -rf $(CU_LIBS_DIR) output
