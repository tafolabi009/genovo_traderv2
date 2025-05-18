import win32serviceutil
import win32service
import win32event
import servicemanager
import socket
import sys
import os
import time
import logging
import subprocess
import psutil
from pathlib import Path

class GenovoTraderService(win32serviceutil.ServiceFramework):
    _svc_name_ = "GenovoTraderV2"
    _svc_display_name_ = "Genovo Trader V2 Service"
    _svc_description_ = "Runs the Genovo Trader V2 bot continuously with automatic recovery"

    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.stop_event = win32event.CreateEvent(None, 0, 0, None)
        self.running = False
        self.process = None
        
        # Setup logging
        log_dir = Path("C:/ProgramData/GenovoTraderV2/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            filename=str(log_dir / "service.log"),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def SvcStop(self):
        """Called when the service is asked to stop"""
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        win32event.SetEvent(self.stop_event)
        self.running = False
        
        # Terminate the bot process if it's running
        if self.process:
            try:
                # Try graceful termination first
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)  # Wait up to 10 seconds
                except subprocess.TimeoutExpired:
                    # If graceful termination fails, force kill
                    self.process.kill()
                logging.info("Bot process terminated")
            except Exception as e:
                logging.error(f"Error terminating bot process: {e}")

    def SvcDoRun(self):
        """Called when the service is asked to start"""
        try:
            self.ReportServiceStatus(win32service.SERVICE_RUNNING)
            logging.info("Service starting...")
            
            # Log to Windows Event Log
            servicemanager.LogMsg(
                servicemanager.EVENTLOG_INFORMATION_TYPE,
                0xF000,  # Custom event ID
                ("Genovo Trader V2 service is starting",)
            )
            
            self.running = True
            self.main()
            
        except Exception as e:
            logging.error(f"Service error: {e}")
            servicemanager.LogErrorMsg(str(e))

    def main(self):
        """Main service loop"""
        # Get the directory where the service script is located
        service_dir = Path(__file__).resolve().parent
        bot_script = service_dir / "main.py"
        
        # Create log file for bot output
        bot_log = Path("C:/ProgramData/GenovoTraderV2/logs/bot_output.log")
        
        # Set up environment variables
        env = os.environ.copy()
        env['PYTHONPATH'] = str(service_dir)  # Add service directory to Python path
        
        # Get the regular Python executable path instead of pythonservice.exe
        python_exe = Path(sys.executable).parent / "python.exe"
        if not python_exe.exists():
            python_exe = Path(sys.executable).parent / "python3.exe"
        if not python_exe.exists():
            logging.error("Could not find Python executable")
            return
            
        logging.info(f"Using Python executable: {python_exe}")
        
        # First, test if Python and imports are working
        logging.info("Testing Python environment...")
        try:
            test_cmd = [
                str(python_exe),
                "-c",
                "import sys; print('Python path:', sys.path); import yaml, pandas, numpy, torch, MetaTrader5; print('All imports successful')"
            ]
            test_process = subprocess.run(
                test_cmd,
                env=env,
                cwd=str(service_dir),
                capture_output=True,
                text=True
            )
            logging.info(f"Test output: {test_process.stdout}")
            if test_process.stderr:
                logging.error(f"Test errors: {test_process.stderr}")
        except Exception as e:
            logging.error(f"Environment test failed: {e}")
        
        while self.running:
            try:
                # Kill any existing MetaTrader processes
                self.cleanup_mt5_processes()
                
                # Start the bot process with output redirection
                logging.info("Starting bot process...")
                with open(bot_log, 'a') as f:
                    f.write(f"\n{'='*50}\n")
                    f.write(f"Starting bot at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Working directory: {service_dir}\n")
                    f.write(f"Python executable: {python_exe}\n")
                    f.write(f"PYTHONPATH: {env.get('PYTHONPATH', 'Not set')}\n")
                    f.write(f"{'='*50}\n")
                
                # First try to run with -v to see import errors
                debug_process = subprocess.run(
                    [str(python_exe), "-v", str(bot_script), "--mode", "live"],
                    cwd=str(service_dir),
                    env=env,
                    capture_output=True,
                    text=True
                )
                with open(bot_log, 'a') as f:
                    f.write("\nDebug Output:\n")
                    f.write(debug_process.stdout)
                    f.write("\nDebug Errors:\n")
                    f.write(debug_process.stderr)
                
                # Now run the actual bot process
                self.process = subprocess.Popen(
                    [str(python_exe), str(bot_script), "--mode", "live"],
                    cwd=str(service_dir),
                    env=env,
                    stdout=open(bot_log, 'a'),
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                )
                
                # Log successful start
                logging.info(f"Bot process started with PID: {self.process.pid}")
                servicemanager.LogMsg(
                    servicemanager.EVENTLOG_INFORMATION_TYPE,
                    0xF001,
                    (f"Bot process started with PID: {self.process.pid}",)
                )
                
                # Monitor the process
                while self.running:
                    if self.process.poll() is not None:
                        # Process has terminated
                        exit_code = self.process.returncode
                        error_msg = f"Bot process terminated with exit code: {exit_code}"
                        logging.warning(error_msg)
                        
                        # Try to get the last few lines of the bot log
                        try:
                            with open(bot_log, 'r') as f:
                                # Read last 10 lines
                                lines = f.readlines()[-10:]
                                error_context = ''.join(lines)
                                logging.warning(f"Last bot output:\n{error_context}")
                        except Exception as e:
                            logging.error(f"Could not read bot log: {e}")
                        
                        servicemanager.LogMsg(
                            servicemanager.EVENTLOG_WARNING_TYPE,
                            0xF002,
                            (error_msg,)
                        )
                        break
                    
                    # Check if we should stop
                    rc = win32event.WaitForSingleObject(self.stop_event, 1000)  # Wait for 1 second
                    if rc == win32event.WAIT_OBJECT_0:
                        break
                
                if not self.running:
                    break
                
                # If we get here, the process died and we should restart
                logging.info("Waiting 60 seconds before restarting bot...")
                time.sleep(60)  # Wait before restart
                
            except Exception as e:
                logging.error(f"Error in main service loop: {e}")
                servicemanager.LogMsg(
                    servicemanager.EVENTLOG_ERROR_TYPE,
                    0xF003,
                    (f"Error in main service loop: {str(e)}",)
                )
                if self.running:
                    time.sleep(60)  # Wait before retry
    
    def cleanup_mt5_processes(self):
        """Kill any existing MetaTrader 5 processes"""
        try:
            mt5_killed = False
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    if proc.info['name'].lower() in ['terminal64.exe', 'metatrader5.exe', 'mt5terminal.exe']:
                        proc_obj = psutil.Process(proc.info['pid'])
                        # Try graceful termination first
                        proc_obj.terminate()
                        try:
                            proc_obj.wait(timeout=10)
                        except psutil.TimeoutExpired:
                            proc_obj.kill()  # Force kill if graceful termination fails
                        mt5_killed = True
                        logging.info(f"Terminated MT5 process: {proc.info['name']} (PID: {proc.info['pid']})")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                
            if mt5_killed:
                # Give extra time for MT5 to fully cleanup if we killed any instances
                time.sleep(10)
            else:
                logging.info("No MT5 processes found to cleanup")
                
        except Exception as e:
            logging.error(f"Error during MT5 process cleanup: {e}")
            # Continue anyway - not critical if cleanup fails

if __name__ == '__main__':
    if len(sys.argv) == 1:
        servicemanager.Initialize()
        servicemanager.PrepareToHostSingle(GenovoTraderService)
        servicemanager.StartServiceCtrlDispatcher()
    else:
        win32serviceutil.HandleCommandLine(GenovoTraderService) 