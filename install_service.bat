@echo off
setlocal

:: Check for admin privileges
net session >nul 2>&1
if %errorLevel% == 0 (
    echo Running with administrator privileges...
) else (
    echo Please run this script as Administrator!
    pause
    exit /b 1
)

:: Install required Python packages
echo Installing required Python packages...
pip install pywin32 psutil

:: Get the directory where this batch file is located
set "SCRIPT_DIR=%~dp0"

:: Create service management menu
:menu
cls
echo.
echo Genovo Trader V2 Service Manager
echo ==============================
echo.
echo 1. Install Service
echo 2. Start Service
echo 3. Stop Service
echo 4. Remove Service
echo 5. View Service Status
echo 6. View Service Logs
echo 7. Exit
echo.
set /p choice="Enter your choice (1-7): "

if "%choice%"=="1" goto install
if "%choice%"=="2" goto start
if "%choice%"=="3" goto stop
if "%choice%"=="4" goto remove
if "%choice%"=="5" goto status
if "%choice%"=="6" goto logs
if "%choice%"=="7" goto end

echo Invalid choice!
timeout /t 2 >nul
goto menu

:install
echo Installing Genovo Trader V2 service...
python "%SCRIPT_DIR%genovo_trader_service.py" install
if %errorLevel% == 0 (
    echo Service installed successfully!
) else (
    echo Failed to install service!
)
pause
goto menu

:start
echo Starting Genovo Trader V2 service...
python "%SCRIPT_DIR%genovo_trader_service.py" start
if %errorLevel% == 0 (
    echo Service started successfully!
) else (
    echo Failed to start service!
)
pause
goto menu

:stop
echo Stopping Genovo Trader V2 service...
python "%SCRIPT_DIR%genovo_trader_service.py" stop
if %errorLevel% == 0 (
    echo Service stopped successfully!
) else (
    echo Failed to stop service!
)
pause
goto menu

:remove
echo Removing Genovo Trader V2 service...
python "%SCRIPT_DIR%genovo_trader_service.py" remove
if %errorLevel% == 0 (
    echo Service removed successfully!
) else (
    echo Failed to remove service!
)
pause
goto menu

:status
echo Checking service status...
sc query GenovoTraderV2
pause
goto menu

:logs
echo Opening service logs...
start "" "C:\ProgramData\GenovoTraderV2\logs\service.log"
pause
goto menu

:end
echo Exiting...
exit /b 0 