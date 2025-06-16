#!/usr/bin/env python3
"""
EMERGENCY DATABASE CLEANUP - AGENT 2
Critical fix for file descriptor leak caused by multiple WAL/SHM files
"""
import os
import sqlite3
import hashlib
import shutil
from pathlib import Path
from datetime import datetime

def emergency_cleanup():
    print("=== EMERGENCY DATABASE CLEANUP - AGENT 2 ===")
    print("Resolving file descriptor leak from meta database WAL/SHM files")
    
    # Create emergency archive
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = f"database_archive_{timestamp}"
    
    try:
        os.makedirs(archive_dir, exist_ok=True)
        print(f"✓ Created archive directory: {archive_dir}")
    except Exception as e:
        archive_dir = "database_archive_emergency"
        os.makedirs(archive_dir, exist_ok=True)
        print(f"✓ Created emergency archive directory: {archive_dir}")
    
    # Get all database files
    current_dir = Path(".")
    db_files = list(current_dir.glob("meta*.db"))
    wal_files = list(current_dir.glob("meta*.db-wal"))
    shm_files = list(current_dir.glob("meta*.db-shm"))
    
    print(f"Found {len(db_files)} database files")
    print(f"Found {len(wal_files)} WAL files (FILE DESCRIPTOR LEAK SOURCE)")
    print(f"Found {len(shm_files)} SHM files (FILE DESCRIPTOR LEAK SOURCE)")
    
    # Backup main database files with checksums
    checksums = {}
    for db_file in db_files:
        try:
            # Calculate checksum
            with open(db_file, 'rb') as f:
                checksum = hashlib.sha256(f.read()).hexdigest()
                checksums[str(db_file)] = checksum
            
            # Create backup
            backup_path = Path(archive_dir) / db_file.name
            shutil.copy2(db_file, backup_path)
            print(f"✓ Backed up {db_file} (SHA256: {checksum[:16]}...)")
        except Exception as e:
            print(f"✗ Error backing up {db_file}: {e}")
    
    # Checkpoint databases to flush WAL files
    checkpoint_success = 0
    for db_file in db_files:
        try:
            conn = sqlite3.connect(str(db_file))
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            conn.execute("PRAGMA wal_checkpoint(RESTART)")
            conn.close()
            checkpoint_success += 1
            print(f"✓ Checkpointed {db_file}")
        except Exception as e:
            print(f"✗ Error checkpointing {db_file}: {e}")
    
    # Remove WAL files (major source of file descriptor leak)
    removed_wal = 0
    for wal_file in wal_files:
        try:
            # Backup first
            backup_path = Path(archive_dir) / wal_file.name
            shutil.copy2(wal_file, backup_path)
            
            # Remove original
            os.remove(wal_file)
            removed_wal += 1
            print(f"✓ Removed WAL file: {wal_file}")
        except Exception as e:
            print(f"✗ Error removing {wal_file}: {e}")
    
    # Remove SHM files
    removed_shm = 0
    for shm_file in shm_files:
        try:
            # Backup first
            backup_path = Path(archive_dir) / shm_file.name
            shutil.copy2(shm_file, backup_path)
            
            # Remove original
            os.remove(shm_file)
            removed_shm += 1
            print(f"✓ Removed SHM file: {shm_file}")
        except Exception as e:
            print(f"✗ Error removing {shm_file}: {e}")
    
    # Verification
    remaining_wal = list(current_dir.glob("meta*.db-wal"))
    remaining_shm = list(current_dir.glob("meta*.db-shm"))
    
    print("\n=== CLEANUP SUMMARY ===")
    print(f"Archive location: {archive_dir}")
    print(f"Database files backed up: {len(checksums)}")
    print(f"Databases checkpointed: {checkpoint_success}/{len(db_files)}")
    print(f"WAL files removed: {removed_wal}/{len(wal_files)}")
    print(f"SHM files removed: {removed_shm}/{len(shm_files)}")
    print(f"Remaining WAL files: {len(remaining_wal)}")
    print(f"Remaining SHM files: {len(remaining_shm)}")
    
    if len(remaining_wal) == 0 and len(remaining_shm) == 0:
        print("✓ FILE DESCRIPTOR LEAK RESOLVED")
    else:
        print("⚠️  Some files may still be present")
    
    # Save checksums
    checksum_file = Path(archive_dir) / "checksums.txt"
    with open(checksum_file, 'w') as f:
        f.write("DATABASE BACKUP CHECKSUMS\n")
        f.write("=" * 50 + "\n")
        for file_path, checksum in checksums.items():
            f.write(f"{file_path}: {checksum}\n")
    
    print(f"✓ Checksums saved to {checksum_file}")
    return archive_dir, checksums, removed_wal, removed_shm

if __name__ == "__main__":
    emergency_cleanup()