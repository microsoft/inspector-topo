## Copyright (c) Microsoft Corporation.
## All rights reserved.

TARGET=topodetect
OBJECTS=topodetect.o Endpoint.o Loopback.o probe.o

# Uncomment to enable debug logging
###DEBUG_LOG=-DDEBUG_LOG

CXXFLAGS+=-std=c++14 -g -MMD -O3 $(DEBUG_LOG)
LDFLAGS+= -L/usr/local/lib \
	-pthread \
	-lnuma \
	-libverbs \
	-lgflags \
	-lhugetlbfs


$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp Makefile
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean::
	rm -f $(TARGET) $(OBJECTS) *.o

#
# automatic dependence stuff
#

realclean:: clean
	rm -rf *.d

deps:: $(OBJECTS)

-include *.d

