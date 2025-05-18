@echo off
setlocal enabledelayedexpansion

:: Check for admin privileges
net session >nul 2>&1
if %errorLevel% == 0 (
    echo Running with administrator privileges...
) else (
    echo Please run this script as Administrator!
    echo Right-click the batch file and select "Run as administrator"
    pause
    exit /b 1
)

:: Get Python path - try multiple common locations
set "PYTHON_PATH="
set "PYTHON_LOCATIONS=C:\Python313\python.exe C:\Python312\python.exe C:\Python311\python.exe C:\Users\Administrator\AppData\Local\Programs\Python\Python313\python.exe C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe C:\Users\Administrator\AppData\Local\Programs\Python\Python311\python.exe"

for %%p in (%PYTHON_LOCATIONS%) do (
    if exist "%%p" (
        set "PYTHON_PATH=%%p"
        goto python_found
    )
)

:: If not found in common locations, try PATH
where python > nul 2>&1
if %errorLevel% == 0 (
    for /f "tokens=*" %%i in ('where python') do (
        :: Skip Windows Store version
        echo %%i | findstr "WindowsApps" > nul
        if errorLevel 1 (
            set "PYTHON_PATH=%%i"
            goto python_found
        )
    )
)

echo Python not found! Please make sure Python is installed.
echo Checked locations:
for %%p in (%PYTHON_LOCATIONS%) do echo - %%p
pause
exit /b 1

:python_found
echo Found Python: %PYTHON_PATH%

:: Install required Python packages
echo Installing required Python packages...
"%PYTHON_PATH%" -m pip install pywin32 psutil

:: Get the directory where this batch file is located
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

:menu
cls
echo.
echo Genovo Trader V2 Service Manager
echo ==============================
echo.
echo 1. Install Service
echo 2. Start Service
echo 3. Stop Service
echo 4. Restart Service
echo 5. Remove Service
echo 6. View Service Status
echo 7. View Service Logs
echo 8. Exit
echo.
set /p choice="Enter your choice (1-8): "

if "%choice%"=="1" goto install
if "%choice%"=="2" goto start
if "%choice%"=="3" goto stop
if "%choice%"=="4" goto restart
if "%choice%"=="5" goto remove
if "%choice%"=="6" goto status
if "%choice%"=="7" goto logs
if "%choice%"=="8" goto end

echo Invalid choice!
timeout /t 2 >nul
goto menu

:install
echo Installing Genovo Trader V2 service...
"%PYTHON_PATH%" genovo_trader_service.py install
if %errorLevel% == 0 (
    echo Service installed successfully!
) else (
    echo Failed to install service!
    echo Please check if Python and all required packages are installed correctly.
    echo Python path: %PYTHON_PATH%
)
pause
goto menu

:start
echo Starting Genovo Trader V2 service...
net start GenovoTraderV2
if %errorLevel% == 0 (
    echo Service started successfully!
) else (
    echo Failed to start service!
)
pause
goto menu

:stop
echo Stopping Genovo Trader V2 service...
net stop GenovoTraderV2
if %errorLevel% == 0 (
    echo Service stopped successfully!
) else (
    echo Failed to stop service!
)
pause
goto menu

:restart
echo Restarting Genovo Trader V2 service...
net stop GenovoTraderV2
timeout /t 5 /nobreak >nul
net start GenovoTraderV2
if %errorLevel% == 0 (
    echo Service restarted successfully!
) else (
    echo Failed to restart service!
)
pause
goto menu

:remove
echo Removing Genovo Trader V2 service...
net stop GenovoTraderV2 2>nul
"%PYTHON_PATH%" genovo_trader_service.py remove
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
if exist "C:\ProgramData\GenovoTraderV2\logs\service.log" (
    start notepad "C:\ProgramData\GenovoTraderV2\logs\service.log"
) else (
    echo Log file not found!
)
if exist "C:\ProgramData\GenovoTraderV2\logs\bot_output.log" (
    start notepad "C:\ProgramData\GenovoTraderV2\logs\bot_output.log"
) else (
    echo Bot output log file not found!
)
pause
goto menu

:end
echo Exiting...
exit /b 0 