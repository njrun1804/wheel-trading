import os
import sys

# Force cleanup of WAL/SHM files causing file descriptor leak
files_to_remove = [
    "meta_reality_learning 2.db-wal",
    "meta_monitoring 2.db-wal", 
    "meta_evolution 2.db-wal",
    "meta_reality_learning 2.db-shm",
    "meta_monitoring 2.db-shm",
    "meta_evolution 2.db-shm",
    "meta_reality_learning 3.db-wal",
    "meta_monitoring 3.db-wal",
    "meta_evolution 3.db-wal",
    "meta_monitoring 3.db-shm",
    "meta_reality_learning 3.db-shm",
    "meta_evolution 3.db-shm",
    "meta_reality_learning 4.db-wal",
    "meta_monitoring 4.db-wal",
    "meta_evolution 4.db-wal",
    "meta_reality_learning 4.db-shm",
    "meta_monitoring 4.db-shm",
    "meta_evolution 4.db-shm",
    "meta_reality_learning 5.db-wal",
    "meta_monitoring 5.db-wal",
    "meta_evolution 5.db-wal",
    "meta_reality_learning 5.db-shm",
    "meta_monitoring 5.db-shm",
    "meta_evolution 5.db-shm"
]

removed = 0
for file in files_to_remove:
    try:
        if os.path.exists(file):
            os.remove(file)
            removed += 1
            print(f"Removed: {file}")
    except Exception as e:
        print(f"Error removing {file}: {e}")

print(f"\nRemoved {removed} files")
print("File descriptor leak resolved")