#!/usr/bin/env python3
"""
Submit a batch download job for 3 years of Unity options data.
This will request Databento to prepare all Unity options data as downloadable files.
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


def submit_unity_batch_job():
    """Submit a batch job for Unity options data."""

    # Initialize client
    logger.info("Initializing Databento client...")
    client = DatabentoClient()

    # Define parameters
    START = "2023-03-28"  # Unity options start date
    END = "2025-06-09"  # Yesterday

    logger.info("=" * 60)
    logger.info("SUBMITTING BATCH JOB FOR UNITY OPTIONS DATA")
    logger.info("=" * 60)
    logger.info(f"Dataset: OPRA.PILLAR")
    logger.info(f"Symbol: U.OPT (all Unity options)")
    logger.info(f"Schema: ohlcv-1d (daily bars)")
    logger.info(f"Date range: {START} to {END}")
    logger.info(f"Delivery: download (via portal)")
    logger.info("=" * 60)

    try:
        # Submit the batch job
        logger.info("Submitting batch job to Databento...")

        job_details = client.client.batch.submit_job(
            dataset="OPRA.PILLAR",
            symbols=["U.OPT"],
            stype_in="parent",
            stype_out="instrument_id",  # Use instrument_id output (supported)
            schema="ohlcv-1d",
            start=START,
            end=END,
            encoding="csv",  # Use CSV to get symbol names
            compression="zstd",  # Best compression
            split_duration="month",  # Split by month for manageable files
            delivery="download",  # Standard download via portal
            map_symbols=True,  # Add symbol column to CSV output
            pretty_px=True,  # Format prices properly
        )

        # Display job details
        logger.info("\n✅ BATCH JOB SUBMITTED SUCCESSFULLY!")
        logger.info("\nJob Details:")
        logger.info(f"  Job ID: {job_details['id']}")
        logger.info(f"  Status: {job_details['state']}")
        logger.info(f"  Dataset: {job_details['dataset']}")
        logger.info(f"  Symbols: {job_details['symbols']}")
        logger.info(f"  Schema: {job_details['schema']}")
        logger.info(f"  Date range: {job_details['start']} to {job_details['end']}")
        logger.info(f"  Encoding: {job_details['encoding']}")
        logger.info(f"  Compression: {job_details['compression']}")
        logger.info(f"  Split by: {job_details['split_duration']}")
        logger.info(f"  Delivery: {job_details['delivery']}")

        logger.info("\n📋 NEXT STEPS:")
        logger.info("1. Monitor job status at: https://databento.com/portal/downloads")
        logger.info(
            f"2. Or check status with: python check_batch_job_status.py {job_details['id']}"
        )
        logger.info("3. Once ready, download files from the portal")
        logger.info("4. Files will be available for 30 days")

        # Estimate size/cost if possible
        logger.info("\n💰 COST ESTIMATE:")
        try:
            # Try to get cost estimate
            cost = client.client.metadata.get_cost(
                dataset="OPRA.PILLAR",
                symbols=["U.OPT"],
                stype_in="parent",
                schema="ohlcv-1d",
                start=START,
                end=END,
                mode="historical",
            )
            logger.info(f"  Estimated cost: ${cost:,.2f}")

            # Get billable size
            size = client.client.metadata.get_billable_size(
                dataset="OPRA.PILLAR",
                symbols=["U.OPT"],
                stype_in="parent",
                schema="ohlcv-1d",
                start=START,
                end=END,
            )
            size_gb = size / (1024**3)
            logger.info(f"  Estimated size: {size_gb:.2f} GB (uncompressed)")
            logger.info(f"  Compressed size will be much smaller with zstd")

        except Exception as e:
            logger.warning(f"Could not estimate cost/size: {e}")

        # Save job ID for later reference
        job_file = Path("unity_options_batch_job.txt")
        with open(job_file, "w") as f:
            f.write(f"Job ID: {job_details['id']}\n")
            f.write(f"Submitted: {datetime.now()}\n")
            f.write(f"Details: {job_details}\n")

        logger.info(f"\nJob ID saved to: {job_file}")

        return job_details

    except Exception as e:
        logger.error(f"Failed to submit batch job: {e}")

        # Check if it's a permissions/subscription issue
        if "subscription" in str(e).lower():
            logger.error("\n❌ SUBSCRIPTION ERROR:")
            logger.error("Your Databento subscription may not include Unity options data.")
            logger.error("Please check your subscription at: https://databento.com/portal/account")

        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    job = submit_unity_batch_job()
