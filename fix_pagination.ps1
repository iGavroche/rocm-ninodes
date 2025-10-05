# RocM Ninodes - Windows Pagination Error Fix Script
# Fixes error 1455: "Le fichier de pagination est insuffisant pour terminer cette opération"

Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "RocM Ninodes - Windows Pagination Error Fix" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""

# Check if running as administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")

if (-not $isAdmin) {
    Write-Host "WARNING: Not running as administrator. Some fixes may not work properly." -ForegroundColor Yellow
    Write-Host "For best results, right-click PowerShell and 'Run as administrator'" -ForegroundColor Yellow
    Write-Host ""
}

# Check system memory
Write-Host "Checking system memory..." -ForegroundColor Green
try {
    $memory = Get-WmiObject -Class Win32_PhysicalMemory | Measure-Object -Property Capacity -Sum
    $totalGB = [math]::Round($memory.Sum / 1GB, 2)
    Write-Host "Total RAM: $totalGB GB" -ForegroundColor White
    
    $availableMemory = Get-WmiObject -Class Win32_OperatingSystem
    $availableGB = [math]::Round($availableMemory.FreePhysicalMemory / 1MB, 2)
    Write-Host "Available RAM: $availableGB GB" -ForegroundColor White
    
    if ($totalGB -lt 16) {
        Write-Host "WARNING: Less than 16GB RAM detected. Paging file increase is highly recommended." -ForegroundColor Red
    }
    
    if ($availableGB -lt 4) {
        Write-Host "CRITICAL: Less than 4GB available memory. Close other applications immediately!" -ForegroundColor Red
    }
} catch {
    Write-Host "Could not check memory information: $($_.Exception.Message)" -ForegroundColor Yellow
}

Write-Host ""

# Apply environment variables
Write-Host "Applying environment variables..." -ForegroundColor Green

# Set environment variables for current session
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True,max_split_size_mb:512"
$env:PYTORCH_HIP_ALLOC_CONF = "expandable_segments:True"
$env:PYTORCH_CUDA_MEMORY_POOL_TYPE = "expandable_segments"

Write-Host "✅ PYTORCH_CUDA_ALLOC_CONF = expandable_segments:True,max_split_size_mb:512" -ForegroundColor Green
Write-Host "✅ PYTORCH_HIP_ALLOC_CONF = expandable_segments:True" -ForegroundColor Green
Write-Host "✅ PYTORCH_CUDA_MEMORY_POOL_TYPE = expandable_segments" -ForegroundColor Green

# Try to set system environment variables (requires admin)
if ($isAdmin) {
    Write-Host ""
    Write-Host "Setting system environment variables (permanent)..." -ForegroundColor Green
    
    try {
        [Environment]::SetEnvironmentVariable("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:512", "User")
        [Environment]::SetEnvironmentVariable("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True", "User")
        [Environment]::SetEnvironmentVariable("PYTORCH_CUDA_MEMORY_POOL_TYPE", "expandable_segments", "User")
        
        Write-Host "✅ System environment variables set permanently" -ForegroundColor Green
    } catch {
        Write-Host "⚠️ Could not set system environment variables: $($_.Exception.Message)" -ForegroundColor Yellow
    }
} else {
    Write-Host "⚠️ Run as administrator to set permanent environment variables" -ForegroundColor Yellow
}

Write-Host ""

# Check if ComfyUI exists
$comfyUIPath = "."
if (Test-Path "main.py") {
    Write-Host "✅ ComfyUI detected in current directory" -ForegroundColor Green
} else {
    Write-Host "⚠️ ComfyUI not found in current directory" -ForegroundColor Yellow
    Write-Host "Please run this script from your ComfyUI directory" -ForegroundColor Yellow
}

Write-Host ""

# Provide manual paging file instructions
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "MANUAL PAGING FILE INCREASE (RECOMMENDED)" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "If you still get pagination errors, manually increase the paging file:" -ForegroundColor Yellow
Write-Host ""
Write-Host "1. Press Win + R, type 'sysdm.cpl', press Enter" -ForegroundColor White
Write-Host "2. Click 'Advanced' tab → 'Performance Settings' → 'Advanced' tab" -ForegroundColor White
Write-Host "3. Under 'Virtual memory', click 'Change'" -ForegroundColor White
Write-Host "4. Uncheck 'Automatically manage paging file size for all drives'" -ForegroundColor White
Write-Host "5. Select your system drive (usually C:)" -ForegroundColor White
Write-Host "6. Select 'Custom size'" -ForegroundColor White
Write-Host "7. Set Initial size: 16384 MB (16 GB)" -ForegroundColor White
Write-Host "8. Set Maximum size: 32768 MB (32 GB)" -ForegroundColor White
Write-Host "9. Click 'Set', then 'OK', then restart ComfyUI" -ForegroundColor White
Write-Host ""

# Check for RocM Ninodes plugin
if (Test-Path "custom_nodes\rocm-ninodes" -or Test-Path "custom_nodes\ComfyUI-ROCM-Optimized-VAE") {
    Write-Host "✅ RocM Ninodes plugin detected" -ForegroundColor Green
} else {
    Write-Host "⚠️ RocM Ninodes plugin not found" -ForegroundColor Yellow
    Write-Host "Install it from: https://github.com/iGavroche/rocm-ninodes" -ForegroundColor Yellow
}

Write-Host ""

# Final instructions
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "NEXT STEPS" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. If you increased the paging file, restart ComfyUI" -ForegroundColor White
Write-Host "2. Use the 'Windows Pagination Diagnostic' node in ComfyUI" -ForegroundColor White
Write-Host "3. If errors persist, try reducing image resolution or batch size" -ForegroundColor White
Write-Host "4. Monitor memory usage in Task Manager during generation" -ForegroundColor White
Write-Host ""

Write-Host "Starting ComfyUI with applied fixes..." -ForegroundColor Green
Write-Host ""

# Start ComfyUI
if (Test-Path "main.py") {
    try {
        python main.py --use-pytorch-cross-attention --highvram --cache-none
    } catch {
        Write-Host "Error starting ComfyUI: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "Try running: python main.py" -ForegroundColor Yellow
    }
} else {
    Write-Host "ComfyUI not found. Please run this script from your ComfyUI directory." -ForegroundColor Red
}

Write-Host ""
Write-Host "Script completed. Press any key to exit..." -ForegroundColor Cyan
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
