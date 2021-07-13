## Copyright (c) Microsoft Corporation.
## Licensed under the MIT License.

TARGET=topodetect
OBJECTS=topodetect.o Endpoint.o Loopback.o probe_latency.o LoopbackFlow.o probe_gpu_bandwidth.o

# Uncomment to enable verbose debug logging
###DEBUG_LOG=-DDEBUG_LOG

NVCC=nvcc
CXXFLAGS+=-std=c++14 -g -MMD -O3 $(DEBUG_LOG)

# Use nvcc for linking.  I replaced -pthread with
# -Xcompiler="-pthread" to get it to work, but I'm not really using
# pthreads so it's probably unnecessary.
LD=nvcc
LDFLAGS+= -L/usr/local/lib \
	-Xcompiler="-pthread" \
	-lnuma \
	-libverbs \
	-lgflags


$(TARGET): $(OBJECTS)
	$(LD) -o $@ $^ $(LDFLAGS)

%.o: %.cpp Makefile
	$(CXX) $(CXXFLAGS) -c -o $@ $<

%.o: %.cu Makefile
	$(NVCC) $(CXXFLAGS) -c -o $@ $<

install:: $(TARGET)
	sudo cp $(TARGET) /usr/local/bin/$(TARGET)

clean::
	rm -f $(TARGET) $(OBJECTS) *.o

#
# automatic dependence stuff
#

realclean:: clean
	rm -rf *.d

deps:: $(OBJECTS)

-include *.d

