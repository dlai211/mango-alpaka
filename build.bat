@echo off
REM Configure and build mango-alpaka.
REM
REM vcvars64 is required: cl.exe is on PATH but the Windows SDK tools
REM (rc.exe, mt.exe) that CMake's compiler check needs are not.

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (
    echo vcvars64.bat failed
    exit /b 1
)

cmake -S "%~dp0." -B "%~dp0build" -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_CXX_COMPILER=cl ^
    -DCMAKE_CUDA_COMPILER=nvcc ^
    -DCMAKE_CUDA_ARCHITECTURES=86 ^
    -DCMAKE_PREFIX_PATH="C:/Users/laiji/.conda/envs/workspace/Library" ^
    -DBoost_ROOT="C:/Users/laiji/.conda/envs/workspace/Library" ^
    -DMANGO_PYTHON="C:/Users/laiji/.conda/envs/workspace/python.exe"
if errorlevel 1 exit /b 1

cmake --build "%~dp0build" --parallel
