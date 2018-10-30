cmake_minimum_required(VERSION 3.1)
project(CTRegression)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
set(CMAKE_INSTALL_PREFIX $ENV{HOME}/bin)
set(CMAKE_CXX_STANDARD 14)#Supported values are 98, 11 and 14.
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic -Wextra")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_BUILD_TYPE Debug)
set(BUILD_SHARED_LIBS False)

#Header directories for the project
set(GLOBAL_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
include_directories(${GLOBAL_INCLUDE_DIR})

#Plog logger
include_directories(${CMAKE_SOURCE_DIR}/submodules/plog/include)

#CLI11 comand line parser library
include_directories(${CMAKE_SOURCE_DIR}/submodules/CLI11/include)

#Matrix class ssh://git@gitlab.stimulate.ovgu.de:2200/robert-frysch/Matrix-Template.git
#include_directories(${CMAKE_SOURCE_DIR}/submodules/Matrix-Template)

#Thread pool management lib ctpl from https://github.com/vit-vit/ctpl
include_directories(${CMAKE_SOURCE_DIR}/submodules/ctpl)
find_package (Threads)#include pthreads

#Intel MKL
find_package(MKL)
include_directories(${MKL_INCLUDE_DIRS})

#CT input output library
FILE( GLOB CTIOL_SRC ${CMAKE_SOURCE_DIR}/submodules/CTIOL/src/*.cpp )
add_library(ctiol ${CTIOL_SRC})
include_directories(${CMAKE_SOURCE_DIR}/submodules/CTIOL/include)

#CTMAL
include_directories(${CMAKE_SOURCE_DIR}/submodules/CTMAL/include)

add_custom_target(formatWebkit
./formatWebkit
WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

SET(PERFVIZ_SOURCES src/perfviz.cpp)
add_executable(PV ${PERFVIZ_SOURCES})
set_target_properties(
    PV 
    PROPERTIES 
        OUTPUT_NAME "perfviz"
        SUFFIX ""
)
#target_link_libraries(PV utils)
target_link_libraries(PV ctiol)
target_link_libraries (PV ${CMAKE_THREAD_LIBS_INIT})
#target_link_libraries(CTRegression /usr/lib/liblapacke.so)
target_link_libraries(PV ${MKL_CORE_LIBRARY})
install (TARGETS PV RUNTIME DESTINATION .)

SET(CONTVIZ_SOURCES src/contviz.cpp)
add_executable(CONTVIZ ${PERFVIZ_SOURCES})
set_target_properties(
    CONTVIZ 
    PROPERTIES 
        OUTPUT_NAME "contviz"
        SUFFIX ""
)
target_link_libraries(CONTVIZ ctiol)
target_link_libraries (CONTVIZ ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(CONTVIZ ${MKL_CORE_LIBRARY})
install (TARGETS CONTVIZ RUNTIME DESTINATION .)

#Documentation target
find_package(Doxygen)
if (DOXYGEN_FOUND)
    # set input and output files
    set(DOXYGEN_IN ${CMAKE_SOURCE_DIR}/doc/doxygen.conf.in)
    set(DOXYGEN_OUT ${CMAKE_SOURCE_DIR}/doc/doxygen.conf)

    # request to configure the file
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT})

    # note the option ALL which allows to build the docs together with the application
    add_custom_target( doxygen_doc
        ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Generating documentation with Doxygen"
        VERBATIM )

    add_custom_target( doxygen
        make
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/doc/latex
        COMMENT "Generating PDF manual"
        VERBATIM )

	add_dependencies(doxygen doxygen_doc)

else (DOXYGEN_FOUND)
  message("Doxygen need to be installed to generate the doxygen documentation")
endif (DOXYGEN_FOUND)

#add_library(Catch INTERFACE)
#target_include_directories(Catch INTERFACE ${CMAKE_SOURCE_DIR}/submodules/Catch2/single_include)
#FILE( GLOB TST_SRC ${CMAKE_SOURCE_DIR}/tests/*.cpp )
#add_executable(test_perfviz ${TST_SRC})
#target_link_libraries(test_ctreg Catch)

#enable_testing()
#add_test(NAME PerfVizTest COMMAND test_perfviz)
