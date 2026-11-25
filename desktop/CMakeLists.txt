cmake_minimum_required(VERSION 3.21)
project(ServeFlowDesktop VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_AUTOMOC ON)
if(MSVC)
    add_compile_options(/utf-8)
endif()
find_package(Qt6 6.5 REQUIRED COMPONENTS Core Network Gui Qml Quick QuickControls2)

add_library(serveflow_client STATIC chatworkspace.cpp chatworkspace.h
    transport.cpp transport.h gateway.cpp gateway.h)
target_include_directories(serveflow_client PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(serveflow_client PUBLIC Qt6::Core Qt6::Network)

qt_add_executable(ServeFlow WIN32 MACOSX_BUNDLE main.cpp)
target_link_libraries(ServeFlow PRIVATE serveflow_client Qt6::Quick Qt6::QuickControls2)
qt_add_resources(ServeFlow desktop_ui PREFIX "/" FILES Main.qml)
set_target_properties(ServeFlow PROPERTIES MACOSX_BUNDLE_GUI_IDENTIFIER "com.zijinglab.serveflow.desktop")

include(CTest)
if(BUILD_TESTING)
    find_package(Qt6 REQUIRED COMPONENTS Test)
    foreach(suite IN ITEMS workspace transport)
        add_executable(tst_${suite} tests/tst_${suite}.cpp)
        target_link_libraries(tst_${suite} PRIVATE serveflow_client Qt6::Test)
        add_test(NAME ${suite} COMMAND tst_${suite})
    endforeach()
endif()

# Optional cross-language integration test uses the backend's existing development dependencies.
set(SERVEFLOW_BACKEND_PYTHON "" CACHE FILEPATH "Backend development Python for the local gateway integration test")
if(BUILD_TESTING AND SERVEFLOW_BACKEND_PYTHON)
    add_executable(tst_window tests/tst_window.cpp)
    qt_add_resources(tst_window test_ui PREFIX "/" FILES Main.qml)
    target_link_libraries(tst_window PRIVATE serveflow_client Qt6::Test Qt6::Quick Qt6::QuickControls2)
    add_test(NAME native_window_fastapi COMMAND "${SERVEFLOW_BACKEND_PYTHON}" -B
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_gateway_flow.py" $<TARGET_FILE:tst_window>)
    set_tests_properties(native_window_fastapi PROPERTIES TIMEOUT 50 ENVIRONMENT
        "QT_QPA_PLATFORM=offscreen;QT_QUICK_BACKEND=software;PYTHON_DOTENV_DISABLED=1")
endif()
