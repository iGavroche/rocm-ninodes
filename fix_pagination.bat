@echo off
REM RocM Ninodes - Windows Pagination Error Fix Script
REM Fixes error 1455: "Le fichier de pagination est insuffisant pour terminer cette opération"

echo ===============================================
echo RocM Ninodes - Windows Pagination Error Fix
echo ===============================================
echo.

echo Checking system memory...
wmic computersystem get TotalPhysicalMemory /value | findstr "TotalPhysicalMemory"
echo.

echo Applying environment variables...
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
set PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
set PYTORCH_CUDA_MEMORY_POOL_TYPE=expandable_segments

echo ^✅ PYTORCH_CUDA_ALLOC_CONF = expandable_segments:True,max_split_size_mb:512
echo ^✅ PYTORCH_HIP_ALLOC_CONF = expandable_segments:True
echo ^✅ PYTORCH_CUDA_MEMORY_POOL_TYPE = expandable_segments
echo.

REM Check if ComfyUI exists
if exist "main.py" (
    echo ^✅ ComfyUI detected in current directory
) else (
    echo ^⚠️ ComfyUI not found in current directory
    echo Please run this script from your ComfyUI directory
    echo.
    pause
    exit /b 1
)

echo.
echo ===============================================
echo MANUAL PAGING FILE INCREASE (RECOMMENDED)
echo ===============================================
echo.
echo If you still get pagination errors, manually increase the paging file:
echo.
echo 1. Press Win + R, type 'sysdm.cpl', press Enter
echo 2. Click 'Advanced' tab → 'Performance Settings' → 'Advanced' tab
echo 3. Under 'Virtual memory', click 'Change'
echo 4. Uncheck 'Automatically manage paging file size for all drives'
echo 5. Select your system drive (usually C:)
echo 6. Select 'Custom size'
echo 7. Set Initial size: 16384 MB (16 GB)
echo 8. Set Maximum size: 32768 MB (32 GB)
echo 9. Click 'Set', then 'OK', then restart ComfyUI
echo.

REM Check for RocM Ninodes plugin
if exist "custom_nodes\rocm-ninodes" (
    echo ^✅ RocM Ninodes plugin detected
) else if exist "custom_nodes\ComfyUI-ROCM-Optimized-VAE" (
    echo ^✅ RocM Ninodes plugin detected
) else (
    echo ^⚠️ RocM Ninodes plugin not found
    echo Install it from: https://github.com/iGavroche/rocm-ninodes
)

echo.
echo ===============================================
echo NEXT STEPS
echo ===============================================
echo.
echo 1. If you increased the paging file, restart ComfyUI
echo 2. Use the 'Windows Pagination Diagnostic' node in ComfyUI
echo 3. If errors persist, try reducing image resolution or batch size
echo 4. Monitor memory usage in Task Manager during generation
echo.

echo Starting ComfyUI with applied fixes...
echo.

REM Start ComfyUI
python main.py --use-pytorch-cross-attention --highvram --cache-none

echo.
echo Script completed. Press any key to exit...
pause >nul
