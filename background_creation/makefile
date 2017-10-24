CC := g++ # This is the main compiler
#CC := clang # and comment out the linker last line for sanity
SRCDIR := source
BUILDDIR := build
TARGET := bin/pmatch
SRCEXT := c
SOURCES := $(shell find $(SRCDIR) -type f -name *.$(SRCEXT))
OBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.$(SRCEXT)=.o))
CFLAGS := -g # -Wall
LIB := -opencv
LDFLAGS := -L /usr/local/Cellar/opencv/3.3.0_3/lib
LDLIBS :=  -lopencv_core -lopencv_imgproc  -lopencv_highgui -lopencv_objdetect -lopencv_imgcodecs
INC := -I include

$(TARGET): $(OBJECTS)
	@echo " Linking..."
	@echo " $(CC) $(LDFLAGS) $(LDLIBS)  $^ -o $(TARGET) $(LIB)"; $(CC) $(LDFLAGS) $^ $(LDLIBS)  $^ -o $(TARGET) $(LIB)

$(BUILDDIR)/%.o: $(SRCDIR)/%.$(SRCEXT)
	@mkdir -p $(BUILDDIR)
	@echo " $(CC) $(CFLAGS) $(INC) -c -o $@ $<"; $(CC) $(CFLAGS) $(INC) -c -o $@ $<

clean:
	@echo " Cleaning..."; 
	@echo " $(RM) -r $(BUILDDIR) $(TARGET)"; $(RM) -r $(BUILDDIR) $(TARGET)
