"""
Google Cloud Storage adapter for cold backup storage.
Minimal implementation focused on cost efficiency.
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from google.cloud import storage
from google.cloud.exceptions import NotFound

from ..utils import get_logger

logger = get_logger(__name__)


@dataclass
class GCSConfig:
    """Configuration for GCS adapter."""

    project_id: str
    raw_bucket: str = "wheel-raw"
    processed_bucket: str = "wheel-processed"

    @classmethod
    def from_env(cls) -> "GCSConfig":
        """Create config from environment variables."""
        return cls(
            project_id=os.environ.get("GCP_PROJECT_ID", ""),
            raw_bucket=os.environ.get("GCS_RAW_BUCKET", "wheel-raw"),
            processed_bucket=os.environ.get("GCS_PROCESSED_BUCKET", "wheel-processed"),
        )


class GCSAdapter:
    """Simple GCS adapter for backup storage."""

    def __init__(self, config: Optional[GCSConfig] = None):
        self.config = config or GCSConfig.from_env()

        # Only initialize if project ID is set
        if self.config.project_id:
            self.client = storage.Client(project=self.config.project_id)
            self.raw_bucket = self.client.bucket(self.config.raw_bucket)
            self.processed_bucket = self.client.bucket(self.config.processed_bucket)
            self.enabled = True
        else:
            self.enabled = False
            logger.info("GCS adapter disabled - no project ID configured")

    async def upload_raw_response(
        self, source: str, timestamp: datetime, data: Dict[str, Any]
    ) -> Optional[str]:
        """Upload raw API response to GCS."""
        if not self.enabled:
            return None

        blob_name = f"{source}/{timestamp:%Y/%m/%d}/{timestamp:%H%M%S}_{source}.json"

        try:
            blob = self.raw_bucket.blob(blob_name)
            blob.upload_from_string(json.dumps(data, default=str), content_type="application/json")

            logger.info("uploaded_raw_to_gcs", source=source, blob=blob_name, size=blob.size)

            return f"gs://{self.config.raw_bucket}/{blob_name}"

        except Exception as e:
            logger.error("gcs_upload_failed", error=str(e), source=source)
            return None

    async def upload_parquet(self, local_file: Path, dataset: str) -> Optional[str]:
        """Upload Parquet file to processed bucket."""
        if not self.enabled:
            return None

        timestamp = datetime.utcnow()
        blob_name = f"{dataset}/year={timestamp.year}/month={timestamp.month:02d}/{local_file.name}"

        try:
            blob = self.processed_bucket.blob(blob_name)
            blob.upload_from_filename(str(local_file))

            logger.info(
                "uploaded_parquet_to_gcs",
                dataset=dataset,
                blob=blob_name,
                size_mb=blob.size / (1024 * 1024),
            )

            return f"gs://{self.config.processed_bucket}/{blob_name}"

        except Exception as e:
            logger.error("gcs_parquet_upload_failed", error=str(e), file=str(local_file))
            return None

    async def download_parquet(self, blob_path: str, local_dir: Path) -> Optional[Path]:
        """Download Parquet file from GCS."""
        if not self.enabled:
            return None

        # Parse bucket and blob name
        if not blob_path.startswith("gs://"):
            return None

        parts = blob_path[5:].split("/", 1)
        if len(parts) != 2:
            return None

        bucket_name, blob_name = parts

        try:
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(blob_name)

            # Create local file path
            local_file = local_dir / Path(blob_name).name
            local_file.parent.mkdir(parents=True, exist_ok=True)

            # Download
            blob.download_to_filename(str(local_file))

            logger.info(
                "downloaded_from_gcs",
                blob=blob_path,
                local=str(local_file),
                size_mb=blob.size / (1024 * 1024),
            )

            return local_file

        except NotFound:
            logger.warning("gcs_blob_not_found", blob=blob_path)
            return None
        except Exception as e:
            logger.error("gcs_download_failed", error=str(e), blob=blob_path)
            return None

    async def list_parquet_files(
        self,
        dataset: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[str]:
        """List available Parquet files for a dataset."""
        if not self.enabled:
            return []

        prefix = f"{dataset}/"
        blobs = []

        try:
            for blob in self.processed_bucket.list_blobs(prefix=prefix):
                # Filter by date if specified
                if start_date or end_date:
                    # Parse year/month from path
                    parts = blob.name.split("/")
                    year_part = next((p for p in parts if p.startswith("year=")), None)
                    month_part = next((p for p in parts if p.startswith("month=")), None)

                    if year_part and month_part:
                        year = int(year_part.split("=")[1])
                        month = int(month_part.split("=")[1])
                        blob_date = datetime(year, month, 1)

                        if start_date and blob_date < start_date:
                            continue
                        if end_date and blob_date > end_date:
                            continue

                blobs.append(f"gs://{self.config.processed_bucket}/{blob.name}")

            logger.info("listed_gcs_files", dataset=dataset, count=len(blobs))

            return blobs

        except Exception as e:
            logger.error("gcs_list_failed", error=str(e), dataset=dataset)
            return []

    def set_lifecycle_policy(self):
        """Set lifecycle policies on buckets."""
        if not self.enabled:
            return

        # Raw bucket: Standard -> Nearline after 30 days, delete after 365
        raw_rules = [
            storage.LifecycleRule(
                action={"type": "SetStorageClass", "storageClass": "NEARLINE"},
                condition={"age": 30},
            ),
            storage.LifecycleRule(action={"type": "Delete"}, condition={"age": 365}),
        ]

        # Processed bucket: Standard -> Nearline after 30 days, delete after 730
        processed_rules = [
            storage.LifecycleRule(
                action={"type": "SetStorageClass", "storageClass": "NEARLINE"},
                condition={"age": 30},
            ),
            storage.LifecycleRule(action={"type": "Delete"}, condition={"age": 730}),
        ]

        try:
            self.raw_bucket.lifecycle_rules = raw_rules
            self.raw_bucket.patch()

            self.processed_bucket.lifecycle_rules = processed_rules
            self.processed_bucket.patch()

            logger.info("gcs_lifecycle_policies_set")

        except Exception as e:
            logger.error("gcs_lifecycle_failed", error=str(e))
