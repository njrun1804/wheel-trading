#!/bin/bash
# Rotate logs daily, keep 30 days

LOG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../logs" && pwd)"
RETENTION_DAYS=30

# Function to rotate a log file
rotate_log() {
    local log_file="$1"
    local base_name=$(basename "$log_file" .log)
    local date_suffix=$(date +%Y%m%d)
    
    if [ -f "$log_file" ]; then
        # Move current log to dated version
        mv "$log_file" "${LOG_DIR}/${base_name}_${date_suffix}.log"
        
        # Compress logs older than 1 day
        find "$LOG_DIR" -name "${base_name}_*.log" -mtime +1 -exec gzip {} \;
        
        # Remove logs older than retention period
        find "$LOG_DIR" -name "${base_name}_*.log.gz" -mtime +$RETENTION_DAYS -delete
        
        # Create new empty log file
        touch "$log_file"
    fi
}

# Rotate each log file
rotate_log "$LOG_DIR/eod_collection.log"
rotate_log "$LOG_DIR/intraday_collection.log"
rotate_log "$LOG_DIR/maintenance.log"

echo "Log rotation completed at $(date)"
