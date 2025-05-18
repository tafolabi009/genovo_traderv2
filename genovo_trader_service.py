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
        
        while self.running:
            try:
                # Kill any existing MetaTrader processes
                self.cleanup_mt5_processes()
                
                # Start the bot process
                logging.info("Starting bot process...")
                self.process = subprocess.Popen(
                    [sys.executable, str(bot_script), "--mode", "live"],
                    cwd=str(service_dir),
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                )
                
                # Log successful start to Windows Event Log
                servicemanager.LogMsg(
                    servicemanager.EVENTLOG_INFORMATION_TYPE,
                    0xF001,  # Custom event ID
                    ("Bot process started successfully",)
                )
                
                # Monitor the process
                while self.running:
                    if self.process.poll() is not None:
                        # Process has terminated
                        exit_code = self.process.returncode
                        logging.warning(f"Bot process terminated with exit code: {exit_code}")
                        # Log termination to Windows Event Log
                        servicemanager.LogMsg(
                            servicemanager.EVENTLOG_WARNING_TYPE,
                            0xF002,  # Custom event ID
                            (f"Bot process terminated with exit code: {exit_code}",)
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
                # Log error to Windows Event Log
                servicemanager.LogMsg(
                    servicemanager.EVENTLOG_ERROR_TYPE,
                    0xF003,  # Custom event ID
                    (f"Error in main service loop: {str(e)}",)
                )
                if self.running:
                    time.sleep(60)  # Wait before retry
    
    def cleanup_mt5_processes(self):
        """Kill any existing MetaTrader 5 processes"""
        try:
            for proc in psutil.process_iter(['pid', 'name']):
                if proc.info['name'] == 'terminal64.exe':
                    try:
                        psutil.Process(proc.info['pid']).terminate()
                        logging.info(f"Terminated MT5 process (PID: {proc.info['pid']})")
                    except Exception as e:
                        logging.error(f"Error terminating MT5 process: {e}")
            time.sleep(5)  # Give processes time to terminate
        except Exception as e:
            logging.error(f"Error during MT5 process cleanup: {e}")

if __name__ == '__main__':
    if len(sys.argv) == 1:
        servicemanager.Initialize()
        servicemanager.PrepareToHostSingle(GenovoTraderService)
        servicemanager.StartServiceCtrlDispatcher()
    else:
        win32serviceutil.HandleCommandLine(GenovoTraderService) 