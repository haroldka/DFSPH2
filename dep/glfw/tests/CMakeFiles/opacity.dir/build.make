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


# Produce verbose output by default.
VERBOSE = 1

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
include dep/glfw/tests/CMakeFiles/opacity.dir/depend.make

# Include the progress variables for this target.
include dep/glfw/tests/CMakeFiles/opacity.dir/progress.make

# Include the compile flags for this target's objects.
include dep/glfw/tests/CMakeFiles/opacity.dir/flags.make

dep/glfw/tests/CMakeFiles/opacity.dir/opacity.c.o: dep/glfw/tests/CMakeFiles/opacity.dir/flags.make
dep/glfw/tests/CMakeFiles/opacity.dir/opacity.c.o: dep/glfw/tests/opacity.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/harold/Desktop/S2/IG3D/Victoire de la nation/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building C object dep/glfw/tests/CMakeFiles/opacity.dir/opacity.c.o"
	cd "/home/harold/Desktop/S2/IG3D/Victoire de la nation/dep/glfw/tests" && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/opacity.dir/opacity.c.o   -c "/home/harold/Desktop/S2/IG3D/Victoire de la nation/dep/glfw/tests/opacity.c"

dep/glfw/tests/CMakeFiles/opacity.dir/opacity.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/opacity.dir/opacity.c.i"
	cd "/home/harold/Desktop/S2/IG3D/Victoire de la nation/dep/glfw/tests" && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E "/home/harold/Desktop/S2/IG3D/Victoire de la nation/dep/glfw/tests/opacity.c" > CMakeFiles/opacity.dir/opacity.c.i

dep/glfw/tests/CMakeFiles/opacity.dir/opacity.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/opacity.dir/opacity.c.s"
	cd "/home/harold/Desktop/S2/IG3D/Victoire de la nation/dep/glfw/tests" && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S "/home/harold/Desktop/S2/IG3D/Victoire de la nation/dep/glfw/tests/opacity.c" -o CMakeFiles/opacity.dir/opacity.c.s

dep/glfw/tests/CMakeFiles/opacity.dir/__/deps/glad_gl.c.o: dep/glfw/tests/CMakeFiles/opacity.dir/flags.make
dep/glfw/tests/CMakeFiles/opacity.dir/__/deps/glad_gl.c.o: dep/glfw/deps/glad_gl.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/harold/Desktop/S2/IG3D/Victoire de la nation/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Building C object dep/glfw/tests/CMakeFiles/opacity.dir/__/deps/glad_gl.c.o"
	cd "/home/harold/Desktop/S2/IG3D/Victoire de la nation/dep/glfw/tests" && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/opacity.dir/__/deps/glad_gl.c.o   -c "/home/harold/Desktop/S2/IG3D/Victoire de la nation/dep/glfw/deps/glad_gl.c"

dep/glfw/tests/CMakeFiles/opacity.dir/__/deps/glad_gl.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/opacity.dir/__/deps/glad_gl.c.i"
	cd "/home/harold/Desktop/S2/IG3D/Victoire de la nation/dep/glfw/tests" && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E "/home/harold/Desktop/S2/IG3D/Victoire de la nation/dep/glfw/deps/glad_gl.c" > CMakeFiles/opacity.dir/__/deps/glad_gl.c.i

dep/glfw/tests/CMakeFiles/opacity.dir/__/deps/glad_gl.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/opacity.dir/__/deps/glad_gl.c.s"
	cd "/home/harold/Desktop/S2/IG3D/Victoire de la nation/dep/glfw/tests" && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S "/home/harold/Desktop/S2/IG3D/Victoire de la nation/dep/glfw/deps/glad_gl.c" -o CMakeFiles/opacity.dir/__/deps/glad_gl.c.s

# Object files for target opacity
opacity_OBJECTS = \
"CMakeFiles/opacity.dir/opacity.c.o" \
"CMakeFiles/opacity.dir/__/deps/glad_gl.c.o"

# External object files for target opacity
opacity_EXTERNAL_OBJECTS =

dep/glfw/tests/opacity: dep/glfw/tests/CMakeFiles/opacity.dir/opacity.c.o
dep/glfw/tests/opacity: dep/glfw/tests/CMakeFiles/opacity.dir/__/deps/glad_gl.c.o
dep/glfw/tests/opacity: dep/glfw/tests/CMakeFiles/opacity.dir/build.make
dep/glfw/tests/opacity: dep/glfw/src/libglfw3.a
dep/glfw/tests/opacity: /usr/lib/x86_64-linux-gnu/libm.so
dep/glfw/tests/opacity: /usr/lib/x86_64-linux-gnu/librt.so
dep/glfw/tests/opacity: /usr/lib/x86_64-linux-gnu/libm.so
dep/glfw/tests/opacity: /usr/lib/x86_64-linux-gnu/libX11.so
dep/glfw/tests/opacity: dep/glfw/tests/CMakeFiles/opacity.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/home/harold/Desktop/S2/IG3D/Victoire de la nation/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_3) "Linking C executable opacity"
	cd "/home/harold/Desktop/S2/IG3D/Victoire de la nation/dep/glfw/tests" && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/opacity.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
dep/glfw/tests/CMakeFiles/opacity.dir/build: dep/glfw/tests/opacity

.PHONY : dep/glfw/tests/CMakeFiles/opacity.dir/build

dep/glfw/tests/CMakeFiles/opacity.dir/clean:
	cd "/home/harold/Desktop/S2/IG3D/Victoire de la nation/dep/glfw/tests" && $(CMAKE_COMMAND) -P CMakeFiles/opacity.dir/cmake_clean.cmake
.PHONY : dep/glfw/tests/CMakeFiles/opacity.dir/clean

dep/glfw/tests/CMakeFiles/opacity.dir/depend:
	cd "/home/harold/Desktop/S2/IG3D/Victoire de la nation" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/home/harold/Desktop/S2/IG3D/Victoire de la nation" "/home/harold/Desktop/S2/IG3D/Victoire de la nation/dep/glfw/tests" "/home/harold/Desktop/S2/IG3D/Victoire de la nation" "/home/harold/Desktop/S2/IG3D/Victoire de la nation/dep/glfw/tests" "/home/harold/Desktop/S2/IG3D/Victoire de la nation/dep/glfw/tests/CMakeFiles/opacity.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : dep/glfw/tests/CMakeFiles/opacity.dir/depend

