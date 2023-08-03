cmake_minimum_required(VERSION 3.1)
project(CTRegression)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
set(CMAKE_INSTALL_PREFIX $ENV{HOME}/bin)
#set(CMAKE_INSTALL_PREFIX /opt/kulvait/bin)
set(CMAKE_CXX_STANDARD 17)#Supported values are 98, 11 and 14.
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic -Wextra")
message("Current value of flags is ${CMAKE_CXX_FLAGS} and debug is ${CMAKE_CXX_FLAGS_DEBUG} and release is ${CMAKE_CXX_FLAGS_RELEASE}")
#Default CMAKE_CXX_FLAGS are empty, default for CMAKE_CXX_FLAGS_DEBUG is -g and CMAKE_CXX_FLAGS_RELEASE are not empty
set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-g -DDEBUG")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic -Wextra")
set(CMAKE_BUILD_TYPE Debug)
set(BUILD_SHARED_LIBS False)
add_definitions("-Dstrtk_no_tr1_or_boost")

#Header directories for the project
set(GLOBAL_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
include_directories(${GLOBAL_INCLUDE_DIR})


#Plog logger
include_directories(${CMAKE_SOURCE_DIR}/submodules/plog/include)

#CLI11 comand line parser library
include_directories(${CMAKE_SOURCE_DIR}/submodules/CLI11/include)

#Matrix class ssh://git@gitlab.stimulate.ovgu.de:2200/robert-frysch/Matrix-Template.git
#include_directories(${CMAKE_SOURCE_DIR}/submodules/Matrix-Template)

#Thread pool management lib ftpl from https://github.com/kulvait/FTPL
include_directories(${CMAKE_SOURCE_DIR}/submodules/FTPL)
find_package (Threads)#include pthreads

#Intel MKL
find_package(MKL)
include_directories(${MKL_INCLUDE_DIRS})

#CT input output library
FILE( GLOB CTIOL_SRC ${CMAKE_SOURCE_DIR}/submodules/CTIOL/src/*.cpp ${CMAKE_SOURCE_DIR}/submodules/CTIOL/src/PROG/*.cpp)
add_library(ctiol ${CTIOL_SRC})
target_link_libraries(ctiol stdc++fs)#include <experimental/filesystem>
include_directories(${CMAKE_SOURCE_DIR}/submodules/CTIOL/include)

#CTMAL
include_directories(${CMAKE_SOURCE_DIR}/submodules/CTMAL/include)

#Matplotlibcpp
include_directories(${CMAKE_SOURCE_DIR}/submodules/matplotlib-cpp)
#Python 3 dependencies
find_package (Python3 COMPONENTS Interpreter Development NumPy)
include_directories(${Python3_INCLUDE_DIRS})
include_directories(${Python3_NumPy_INCLUDE_DIRS})

#Git versioning
INCLUDE(${CMAKE_SOURCE_DIR}/submodules/gitversion/cmake.cmake)
include_directories(${CMAKE_SOURCE_DIR}/submodules/gitversion)

add_custom_target(formatWebkit
./formatWebkit
WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

SET(PERFVIZ_TIMESERIES_SOURCES src/perfviz-timeseries.cpp src/PerfusionVizualizationArguments.cpp)
add_executable(perfvizTimeseries ${PERFVIZ_TIMESERIES_SOURCES})
set_target_properties(
    perfvizTimeseries
    PROPERTIES 
        OUTPUT_NAME "perfviz-timeseries"
        SUFFIX ""
)
target_git_version_init(perfvizTimeseries)
target_link_libraries(perfvizTimeseries ctiol)
target_link_libraries(perfvizTimeseries ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(perfvizTimeseries ${MKL_CORE_LIBRARY})
target_link_libraries(perfvizTimeseries ${Python3_LIBRARIES})
install (TARGETS perfvizTimeseries RUNTIME DESTINATION .)

SET(PERFVIZ_POLYNOMIAL_SOURCES src/perfviz-polynomial.cpp src/PerfusionVizualizationArguments.cpp)
add_executable(perfvizPolynomial ${PERFVIZ_POLYNOMIAL_SOURCES})
set_target_properties(
    perfvizPolynomial
    PROPERTIES 
        OUTPUT_NAME "perfviz-polynomial"
        SUFFIX ""
)
target_git_version_init(perfvizPolynomial)
target_link_libraries(perfvizPolynomial ctiol)
target_link_libraries(perfvizPolynomial ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(perfvizPolynomial ${MKL_CORE_LIBRARY})
target_link_libraries(perfvizPolynomial ${Python3_LIBRARIES})
install (TARGETS perfvizPolynomial RUNTIME DESTINATION .)

SET(PERFVIZ_FOURIER_SOURCES src/perfviz-fourier.cpp src/PerfusionVizualizationArguments.cpp)
add_executable(perfvizFourier ${PERFVIZ_FOURIER_SOURCES})
set_target_properties(
    perfvizFourier 
    PROPERTIES 
        OUTPUT_NAME "perfviz-fourier"
        SUFFIX ""
)
target_git_version_init(perfvizFourier)
target_link_libraries(perfvizFourier ctiol)
target_link_libraries (perfvizFourier ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(perfvizFourier ${MKL_CORE_LIBRARY})
target_link_libraries(perfvizFourier ${Python3_LIBRARIES})
install (TARGETS perfvizFourier RUNTIME DESTINATION .)

SET(PERFVIZ_FOURIER_KVA_SOURCES src/perfviz-fourier-kva.cpp src/PerfusionVizualizationArguments.cpp)
add_executable(perfvizFourierKVA ${PERFVIZ_FOURIER_KVA_SOURCES})
set_target_properties(
    perfvizFourierKVA 
    PROPERTIES 
        OUTPUT_NAME "perfviz-fourier-kva"
        SUFFIX ""
)
target_git_version_init(perfvizFourierKVA)
target_link_libraries(perfvizFourierKVA ctiol)
target_link_libraries (perfvizFourierKVA ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(perfvizFourierKVA ${MKL_CORE_LIBRARY})
target_link_libraries(perfvizFourierKVA ${Python3_LIBRARIES})
install (TARGETS perfvizFourierKVA RUNTIME DESTINATION .)

SET(PERFVIZXXX_SOURCES src/perfviz-engineer.cpp src/PerfusionVizualizationArguments.cpp)
add_executable(perfviz_engineer ${PERFVIZXXX_SOURCES})
set_target_properties(
    perfviz_engineer 
    PROPERTIES 
        OUTPUT_NAME "perfviz-engineer"
        SUFFIX ""
)
target_git_version_init(perfviz_engineer)
target_link_libraries(perfviz_engineer ctiol)
target_link_libraries (perfviz_engineer ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(perfviz_engineer ${MKL_CORE_LIBRARY})
target_link_libraries(perfviz_engineer ${Python3_LIBRARIES})
install (TARGETS perfviz_engineer RUNTIME DESTINATION .)

SET(PERFVIZSTATICREC_SOURCES src/perfviz-staticrec.cpp src/PerfusionVizualizationArguments.cpp)
add_executable(perfviz_staticrec ${PERFVIZSTATICREC_SOURCES})
set_target_properties(
    perfviz_staticrec 
    PROPERTIES 
        OUTPUT_NAME "perfviz-staticrec"
        SUFFIX ""
)
target_git_version_init(perfviz_staticrec)
target_link_libraries(perfviz_staticrec ctiol)
target_link_libraries (perfviz_staticrec ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(perfviz_staticrec ${MKL_CORE_LIBRARY})
target_link_libraries(perfviz_staticrec ${Python3_LIBRARIES})
install (TARGETS perfviz_staticrec RUNTIME DESTINATION .)

SET(PERFVIZCTREC_SOURCES src/perfviz-ctrec.cpp src/PerfusionVizualizationArguments.cpp)
add_executable(perfviz_ctrec ${PERFVIZCTREC_SOURCES})
set_target_properties(
    perfviz_ctrec 
    PROPERTIES 
        OUTPUT_NAME "perfviz-ctrec"
        SUFFIX ""
)
target_git_version_init(perfviz_ctrec)
target_link_libraries(perfviz_ctrec ctiol)
target_link_libraries (perfviz_ctrec ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(perfviz_ctrec ${MKL_CORE_LIBRARY})
target_link_libraries(perfviz_ctrec ${Python3_LIBRARIES})
install (TARGETS perfviz_ctrec RUNTIME DESTINATION .)

SET(PERFVIZCTVOL_SOURCES src/perfviz-ctvol.cpp src/PerfusionVizualizationArguments.cpp)
add_executable(perfviz_ctvol ${PERFVIZCTVOL_SOURCES})
set_target_properties(
    perfviz_ctvol 
    PROPERTIES 
        OUTPUT_NAME "perfviz-ctvol"
        SUFFIX ""
)
target_git_version_init(perfviz_ctvol)
target_link_libraries(perfviz_ctvol ctiol)
target_link_libraries (perfviz_ctvol ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(perfviz_ctvol ${MKL_CORE_LIBRARY})
target_link_libraries(perfviz_ctvol ${Python3_LIBRARIES})
install (TARGETS perfviz_ctvol RUNTIME DESTINATION .)

SET(PERFVIZCTKVA_SOURCES src/perfviz-ctkva.cpp src/PerfusionVizualizationArguments.cpp)
add_executable(perfviz_ctkva ${PERFVIZCTKVA_SOURCES})
set_target_properties(
    perfviz_ctkva 
    PROPERTIES 
        OUTPUT_NAME "perfviz-ctkva"
        SUFFIX ""
)
target_git_version_init(perfviz_ctkva)
target_link_libraries(perfviz_ctkva ctiol)
target_link_libraries (perfviz_ctkva ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(perfviz_ctkva ${MKL_CORE_LIBRARY})
target_link_libraries(perfviz_ctkva ${Python3_LIBRARIES})
install (TARGETS perfviz_ctkva RUNTIME DESTINATION .)

SET(CONTVIZ_SOURCES src/contviz.cpp src/PerfusionVizualizationArguments.cpp)
add_executable(contviz ${CONTVIZ_SOURCES})
set_target_properties(
    contviz 
    PROPERTIES 
        OUTPUT_NAME "contviz"
        SUFFIX ""
)
target_link_libraries(contviz ctiol)
target_link_libraries (contviz ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(contviz ${MKL_CORE_LIBRARY})
target_link_libraries(contviz ${Python3_LIBRARIES})
install (TARGETS contviz RUNTIME DESTINATION .)

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
