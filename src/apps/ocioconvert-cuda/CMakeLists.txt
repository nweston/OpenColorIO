if (OIIO_FOUND)
    include_directories(
        ${CMAKE_SOURCE_DIR}/export/
        ${CMAKE_BINARY_DIR}/export/
        ${CMAKE_SOURCE_DIR}/src/apps/share/
        ${OIIO_INCLUDES}
    )

    file(GLOB_RECURSE share_src_files "${CMAKE_SOURCE_DIR}/src/apps/share/*.cpp")

    cuda_add_executable(ocioconvert-cuda ${share_src_files} main.cpp)

    target_link_libraries(ocioconvert-cuda ${OIIO_LIBRARIES} OpenColorIO dl)

    install(TARGETS ocioconvert-cuda DESTINATION ${CMAKE_INSTALL_EXEC_PREFIX}/bin)
endif()
