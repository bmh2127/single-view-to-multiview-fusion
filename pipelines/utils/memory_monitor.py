"""
Memory monitoring utility for tracking memory usage during training.

This module provides a simple thread-based memory monitor to track both system
and process memory usage during model training.
"""

import os
import psutil
import time
import threading
import logging
from datetime import datetime

class MemoryMonitor:
    """Monitor system and process memory usage during training."""
    
    def __init__(self, interval=5.0, log_file=None):
        """
        Initialize the memory monitor.
        
        Args:
            interval: Time interval in seconds between memory checks
            log_file: Optional file to log memory stats
        """
        self.interval = interval
        self.running = False
        self.thread = None
        self.log_file = log_file
        self._setup_logging()
    
    def _setup_logging(self):
        """Set up logging configuration."""
        self.logger = logging.getLogger("MemoryMonitor")
        self.logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if self.log_file:
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        process = psutil.Process(os.getpid())
        
        while self.running:
            # Get system memory stats
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Get process memory info
            process_memory = process.memory_info()
            
            # Log stats
            self.logger.info(f"Process: RSS={process_memory.rss / (1024**3):.2f} GB, "
                            f"VMS={process_memory.vms / (1024**3):.2f} GB | "
                            f"System: {memory.used / (1024**3):.2f}/{memory.total / (1024**3):.2f} GB "
                            f"({memory.percent:.1f}%) | "
                            f"Swap: {swap.used / (1024**3):.2f}/{swap.total / (1024**3):.2f} GB")
            
            time.sleep(self.interval)
    
    def start(self):
        """Start monitoring memory usage."""
        if self.running:
            self.logger.warning("Memory monitor is already running")
            return
        
        self.logger.info("Starting memory monitoring")
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        """Stop monitoring memory usage."""
        if not self.running:
            self.logger.warning("Memory monitor is not running")
            return
        
        self.logger.info("Stopping memory monitoring")
        self.running = False
        if self.thread:
            self.thread.join(timeout=self.interval + 1.0)


def start_monitoring(interval=5.0, log_file=None):
    """
    Start memory monitoring in a separate thread.
    
    Args:
        interval: Time interval in seconds between memory checks
        log_file: Optional file to log memory stats
        
    Returns:
        MemoryMonitor instance
    """
    monitor = MemoryMonitor(interval=interval, log_file=log_file)
    monitor.start()
    return monitor


if __name__ == "__main__":
    """Run the memory monitor as a standalone script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor system memory usage")
    parser.add_argument("--interval", type=float, default=5.0,
                        help="Time interval in seconds between memory checks")
    parser.add_argument("--log-file", type=str, default=None,
                        help="File to log memory stats")
    parser.add_argument("--duration", type=float, default=60.0,
                        help="Duration in seconds to run the monitor")
    
    args = parser.parse_args()
    
    # Create and start monitor
    monitor = MemoryMonitor(interval=args.interval, log_file=args.log_file)
    monitor.start()
    
    try:
        print(f"Monitoring for {args.duration} seconds... (Ctrl+C to stop)")
        time.sleep(args.duration)
    except KeyboardInterrupt:
        print("Monitoring interrupted by user")
    finally:
        monitor.stop()