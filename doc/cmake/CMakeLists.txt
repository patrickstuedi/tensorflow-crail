PROJECT(crail-tensorflow)
cmake_minimum_required(VERSION 2.6)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CRAIL_INC "/usr/local/include" CACHE STRING "Crail include directory")
set(CRAIL_LIB "/usr/local/lib" CACHE STRING "Crail library directory")


execute_process(COMMAND python -c "import tensorflow as tf; print(''.join(tf.sysconfig.get_include()))" OUTPUT_VARIABLE TF_INC OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND python -c "import tensorflow as tf; print(''.join(tf.sysconfig.get_lib()))" OUTPUT_VARIABLE TF_LIB OUTPUT_STRIP_TRAILING_WHITESPACE)

#message("CRAIL_INC ${CRAIL_INC}")
#message("CRAIL_LIB ${CRAIL_LIB}")

include_directories(${TF_INC})
include_directories(${CRAIL_INC})
link_directories(${TF_LIB})
link_directories(${CRAIL_LIB})

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")

add_library (crailtensorflow SHARED 
	crail_dataset.cc
	)

target_link_libraries(crailtensorflow tensorflow_framework crail)

install(TARGETS crailtensorflow DESTINATION /lib)
install (FILES 
	crail_dataset.h
	DESTINATION /include)

