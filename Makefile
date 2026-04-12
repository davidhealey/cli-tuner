CXX      := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter

# Use pkg-config when available, fall back to plain -l flags
PKG_SNDFILE    := $(shell pkg-config --cflags --libs sndfile    2>/dev/null || echo "-lsndfile")
PKG_SAMPLERATE := $(shell pkg-config --cflags --libs samplerate 2>/dev/null || echo "-lsamplerate")
PKG_AUBIO      := $(shell pkg-config --cflags --libs aubio      2>/dev/null || echo "-laubio")

LDFLAGS  := $(PKG_SNDFILE) $(PKG_SAMPLERATE) $(PKG_AUBIO)

SRCDIR   := src
SRCS     := $(SRCDIR)/main.cpp $(SRCDIR)/audio.cpp
OBJS     := $(SRCS:.cpp=.o)
TARGET   := cli-tuner

.PHONY: all clean install

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS)

$(SRCDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) -I$(SRCDIR) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

install: $(TARGET)
	install -m 755 $(TARGET) /usr/local/bin/

# Header dependencies
$(SRCDIR)/main.o:  $(SRCDIR)/main.cpp  $(SRCDIR)/audio.h
$(SRCDIR)/audio.o: $(SRCDIR)/audio.cpp $(SRCDIR)/audio.h
