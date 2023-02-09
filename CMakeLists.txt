cmake_minimum_required(VERSION 3.10)

include(RezBuild)

FILE(GLOB_RECURSE py_files "python/*.py")

rez_install_dirs(
    python
    DESTINATION .
    LOCAL_SYMLINK
)

rez_install_dirs(
    nuke/gizmos
    DESTINATION .
)

rez_install_dirs(
    bin
    DESTINATION .
    EXECUTABLE
)

rez_install_dirs(
    environments
    DESTINATION .
    LOCAL_SYMLINK
)

rez_install_dirs(
    models
    DESTINATION .
    LOCAL_SYMLINK
)


add_subdirectory(nuke)