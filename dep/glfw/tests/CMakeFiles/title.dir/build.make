# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/home/harold/Desktop/S2/IG3D/Victoire de la nation"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/home/harold/Desktop/S2/IG3D/Victoire de la nation"

# Include any dependencies generated for this target.
include dep/glfw/tests/CMakeFiles/title.dir/depend.make

# Include the progress variables for this target.
include dep/glfw/tests/CMakeFiles/title.dir/progress.make

# Include the compile flags for this target's objects.
include dep/glfw/tests/CMakeFiles/title.dir/flags.make

dep/glfw/tests/CMakeFiles/title.dir/title.c.o: dep/glfw/tests/CMakeFiles/title.dir/flags.make
dep/glfw/tests/CMakeFiles/title.dir/title.c.o: dep/glfw/tests/title.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/harold/Desktop/S2/IG3D/Victoire de la nation/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building C object dep/glfw/tests/CMakeFiles/title.dir/title.c.o"
	cd "/home/harold/Desktop/S2/IG3D/Victoire de la nation/dep/glfw/tests" && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/title.dir/title.c.o   -c "/home/harold/Desktop/S2/IG3D/Victoire de la nation/dep/glfw/tests/title.c"

dep/glfw/tests/CMakeFiles/title.dir/title.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/title.dir/title.c.i"
	cd "/home/harold/Desktop/S2/IG3D/Victoire de la nation/dep/glfw/tests" && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E "/home/harold/Desktop/S2/IG3D/Victoire de la nation/dep/glfw/tests/title.c" > CMakeFiles/title.dir/title.c.i

dep/glfw/tests/CMakeFiles/title.dir/title.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/title.dir/title.c.s"
	cd "/home/harold/Desktop/S2/IG3D/Victoire de la nation/dep/glfw/tests" && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S "/home/harold/Desktop/S2/IG3D/Victoire de la nation/dep/glfw/tests/title.c" -o CMakeFiles/title.dir/title.c.s

dep/glfw/tests/CMakeFiles/title.dir/__/deps/glad_gl.c.o: dep/glfw/tests/CMakeFiles/title.dir/flags.make
dep/glfw/tests/CMakeFiles/title.dir/__/deps/glad_gl.c.o: dep/glfw/deps/glad_gl.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/harold/Desktop/S2/IG3D/Victoire de la nation/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Building C object dep/glfw/tests/CMakeFiles/title.dir/__/deps/glad_gl.c.o"
	cd "/home/harold/Desktop/S2/IG3D/Victoire de la nation/dep/glfw/tests" && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/title.dir/__/deps/glad_gl.c.o   -c "/home/harold/Desktop/S2/IG3D/Victoire de la nation/dep/glfw/deps/glad_gl.c"

dep/glfw/tests/CMakeFiles/title.dir/__/deps/glad_gl.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/title.dir/__/deps/glad_gl.c.i"
	cd "/home/harold/Desktop/S2/IG3D/Victoire de la nation/dep/glfw/tests" && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E "/home/harold/Desktop/S2/IG3D/Victoire de la nation/dep/glfw/deps/glad_gl.c" > CMakeFiles/title.dir/__/deps/glad_gl.c.i

dep/glfw/tests/CMakeFiles/title.dir/__/deps/glad_gl.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/title.dir/__/deps/glad_gl.c.s"
	cd "/home/harold/Desktop/S2/IG3D/Victoire de la nation/dep/glfw/tests" && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S "/home/harold/Desktop/S2/IG3D/Victoire de la nation/dep/glfw/deps/glad_gl.c" -o CMakeFiles/title.dir/__/deps/glad_gl.c.s

# Object files for target title
title_OBJECTS = \
"CMakeFiles/title.dir/title.c.o" \
"CMakeFiles/title.dir/__/deps/glad_gl.c.o"

# External object files for target title
title_EXTERNAL_OBJECTS =

dep/glfw/tests/title: dep/glfw/tests/CMakeFiles/title.dir/title.c.o
dep/glfw/tests/title: dep/glfw/tests/CMakeFiles/title.dir/__/deps/glad_gl.c.o
dep/glfw/tests/title: dep/glfw/tests/CMakeFiles/title.dir/build.make
dep/glfw/tests/title: dep/glfw/src/libglfw3.a
dep/glfw/tests/title: /usr/lib/x86_64-linux-gnu/libm.so
dep/glfw/tests/title: /usr/lib/x86_64-linux-gnu/librt.so
dep/glfw/tests/title: /usr/lib/x86_64-linux-gnu/libm.so
dep/glfw/tests/title: /usr/lib/x86_64-linux-gnu/libX11.so
dep/glfw/tests/title: dep/glfw/tests/CMakeFiles/title.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/home/harold/Desktop/S2/IG3D/Victoire de la nation/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_3) "Linking C executable title"
	cd "/home/harold/Desktop/S2/IG3D/Victoire de la nation/dep/glfw/tests" && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/title.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
dep/glfw/tests/CMakeFiles/title.dir/build: dep/glfw/tests/title

.PHONY : dep/glfw/tests/CMakeFiles/title.dir/build

dep/glfw/tests/CMakeFiles/title.dir/clean:
	cd "/home/harold/Desktop/S2/IG3D/Victoire de la nation/dep/glfw/tests" && $(CMAKE_COMMAND) -P CMakeFiles/title.dir/cmake_clean.cmake
.PHONY : dep/glfw/tests/CMakeFiles/title.dir/clean

dep/glfw/tests/CMakeFiles/title.dir/depend:
	cd "/home/harold/Desktop/S2/IG3D/Victoire de la nation" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/home/harold/Desktop/S2/IG3D/Victoire de la nation" "/home/harold/Desktop/S2/IG3D/Victoire de la nation/dep/glfw/tests" "/home/harold/Desktop/S2/IG3D/Victoire de la nation" "/home/harold/Desktop/S2/IG3D/Victoire de la nation/dep/glfw/tests" "/home/harold/Desktop/S2/IG3D/Victoire de la nation/dep/glfw/tests/CMakeFiles/title.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : dep/glfw/tests/CMakeFiles/title.dir/depend

