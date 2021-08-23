add_executable(test_trt_crnn test.cpp)
target_link_libraries(test_trt_crnn ${TRT_CRNN_LIBRARY_NAME})
set(OPENCV_NEED_LIBS opencv_imgproc opencv_highgui)
target_link_libraries(test_trt_crnn ${OPENCV_NEED_LIBS})
