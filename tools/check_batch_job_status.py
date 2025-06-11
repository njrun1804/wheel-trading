#!/usr/bin/env python3
"""
Check the status of a Databento batch job and download when ready.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.unity_wheel.data_providers.databento import DatabentoClient

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


def check_job_status(job_id: str = None):
    """Check status of a batch job."""

    # Initialize client
    client = DatabentoClient()

    # If no job ID provided, try to read from file
    if not job_id:
        job_file = Path("unity_options_batch_job.txt")
        if job_file.exists():
            with open(job_file, "r") as f:
                first_line = f.readline()
                if first_line.startswith("Job ID:"):
                    job_id = first_line.split(": ")[1].strip()

        if not job_id:
            logger.error("No job ID provided and no saved job found.")
            logger.info("Usage: python check_batch_job_status.py [JOB_ID]")
            return

    logger.info(f"Checking status for job: {job_id}")
    logger.info("=" * 60)

    try:
        # Get all jobs and find ours
        jobs = client.client.batch.list_jobs(states=["received", "queued", "processing", "done"])

        our_job = None
        for job in jobs:
            if job["id"] == job_id:
                our_job = job
                break

        if not our_job:
            logger.error(f"Job {job_id} not found in active jobs.")
            logger.info("It may have expired (jobs expire after 30 days).")
            return

        # Display job status
        logger.info(f"Job ID: {our_job['id']}")
        logger.info(f"Status: {our_job['state']}")
        logger.info(f"Dataset: {our_job['dataset']}")
        logger.info(f"Date range: {our_job['start']} to {our_job['end']}")

        # Show progress if available
        if "progress" in our_job:
            logger.info(f"Progress: {our_job['progress']}%")

        # Show timing information
        if our_job["ts_received"]:
            logger.info(f"Received: {our_job['ts_received']}")
        if our_job["ts_queued"]:
            logger.info(f"Queued: {our_job['ts_queued']}")
        if our_job["ts_process_start"]:
            logger.info(f"Processing started: {our_job['ts_process_start']}")
        if our_job["ts_process_done"]:
            logger.info(f"Processing completed: {our_job['ts_process_done']}")

        # If done, show details
        if our_job["state"] == "done":
            logger.info("\n✅ JOB COMPLETE!")

            if our_job["record_count"]:
                logger.info(f"Records: {our_job['record_count']:,}")
            if our_job["cost_usd"]:
                logger.info(f"Cost: ${our_job['cost_usd']:,.2f}")
            if our_job["actual_size"]:
                size_gb = our_job["actual_size"] / (1024**3)
                logger.info(f"Compressed size: {size_gb:.2f} GB")
            if our_job["ts_expiration"]:
                logger.info(f"Expires: {our_job['ts_expiration']}")

            # List available files
            logger.info("\n📁 AVAILABLE FILES:")
            files = client.client.batch.list_files(job_id)

            total_size = 0
            data_files = []

            for file in files:
                size_mb = file["size"] / (1024**2)
                logger.info(f"  {file['filename']} ({size_mb:.2f} MB)")
                total_size += file["size"]

                # Collect data files (not metadata)
                if not file["filename"].endswith(".json"):
                    data_files.append(file["filename"])

            total_gb = total_size / (1024**3)
            logger.info(f"\nTotal download size: {total_gb:.2f} GB")

            # Offer to download
            logger.info("\n📥 DOWNLOAD OPTIONS:")
            logger.info("1. Download from portal: https://databento.com/portal/downloads")
            logger.info(f"2. Download programmatically:")
            logger.info(f"   python download_batch_files.py {job_id}")
            logger.info(f"3. Download specific file:")
            if data_files:
                logger.info(f"   python download_batch_files.py {job_id} {data_files[0]}")

        elif our_job["state"] == "processing":
            logger.info("\n⏳ Job is currently processing...")
            logger.info("Check back later or monitor at: https://databento.com/portal/downloads")

        elif our_job["state"] == "queued":
            logger.info("\n⏳ Job is queued for processing...")
            logger.info("This may take some time depending on system load.")

        elif our_job["state"] == "received":
            logger.info("\n⏳ Job has been received and will be queued soon...")

    except Exception as e:
        logger.error(f"Error checking job status: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # Check if job ID provided as argument
    job_id = sys.argv[1] if len(sys.argv) > 1 else None
    check_job_status(job_id)
