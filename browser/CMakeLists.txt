cmake_minimum_required(VERSION 3.20)
project(NebulaBrowser VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Options ---
option(NEBULA_TOR "Enable Tor integration" ON)
option(NEBULA_ADBLOCK "Enable ad blocker" ON)
option(NEBULA_SHIELD "Enable VoidShield malware scanning" ON)
option(NEBULA_FRONTENDS "Enable privacy frontend redirects" ON)
option(NEBULA_SANDBOX "Enable process sandboxing" ON)

# --- C++ Sources (Core browser, UI, frontend registry) ---
set(CPP_SOURCES
    core/nebula.cpp
    frontends/registry.cpp
)

add_executable(nebula-browser ${CPP_SOURCES})
target_include_directories(nebula-browser PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# --- Rust Library (via corrosion or manual cargo) ---
# Security-critical components: adblock, tor, shield, auth, sandbox
find_program(CARGO cargo)
if(CARGO)
    add_custom_target(nebula-rust
        COMMAND ${CARGO} build --release --manifest-path ${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Building Rust security components..."
    )
    add_dependencies(nebula-browser nebula-rust)

    # Link Rust static library
    target_link_directories(nebula-browser PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/target/release
    )
    target_link_libraries(nebula-browser PRIVATE nebula)
endif()

# --- Zig Library (via zig build or manual) ---
find_program(ZIG zig)
if(ZIG)
    add_custom_target(nebula-zig
        COMMAND ${ZIG} build -Doptimize=ReleaseFast
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Building Zig performance components..."
    )
    add_dependencies(nebula-browser nebula-zig)
endif()


# --- Compile flags ---
target_compile_options(nebula-browser PRIVATE
    -Wall -Wextra -Wpedantic
    -O3 -march=native
    -fno-exceptions  # Performance: no exception overhead
    -fno-rtti        # Performance: no RTTI overhead
)

# --- Link flags ---
target_link_options(nebula-browser PRIVATE
    -flto            # Link-time optimization
    -Wl,-z,now       # Full RELRO
    -Wl,-z,relro
    -Wl,-z,noexecstack
)

# --- Sanitizer builds ---
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(nebula-browser PRIVATE
        -fsanitize=address,undefined
        -fno-omit-frame-pointer
    )
    target_link_options(nebula-browser PRIVATE
        -fsanitize=address,undefined
    )
endif()

# --- Install ---
install(TARGETS nebula-browser RUNTIME DESTINATION bin)
