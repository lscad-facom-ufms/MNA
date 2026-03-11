# DI-MNA

Uses the DI-MNA algorithm to allocate jobs in a IoT network

------------------------------------------------------------------------

## Requirements

Before compiling the project, make sure you have the following
installed:

-   **CMake** (version 3.21 or newer recommended)
-   **C++ compiler** compatible with your platform
-   **vcpkg** for dependency management

### vcpkg Setup

This project uses **vcpkg** to manage external dependencies.

You must configure the `VCPKG_ROOT` environment variable inside
**`CMakeUserPresets.json`** so that CMake can locate your vcpkg
installation.

Example:

``` json
{
  "version": 2,
  "configurePresets": [
    {
      "name": "default",
      "cacheVariables": {
        "CMAKE_TOOLCHAIN_FILE": "$env{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
      }
    }
  ]
}
```

Make sure that `VCPKG_ROOT` points to the root directory of your
**vcpkg** installation.

------------------------------------------------------------------------

## Building

To configure and compile the project, run the following commands:

``` bash
cmake --preset=default
cmake --build build
```

After compilation, the executable will be generated inside the `build`
directory.

------------------------------------------------------------------------

## Running

To run the generated binary, the following command line arguments must
be provided:

  Argument   Description
  ---------- ----------------------
  `--in`     Input directory
  `--out`    Output directory
  `--runs`   Number of executions
  `--cs`     cut_sol (alpha parameter)
  `--cs`     cut_comb_nodes (beta parameter)

### Example

``` bash
./program --in input_dir --out output_dir --runs 10 --ccn 500 --cs 2
```

------------------------------------------------------------------------
