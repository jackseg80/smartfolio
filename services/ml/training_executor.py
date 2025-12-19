"""
ML Training Executor Service

Manages ML model training jobs triggered from Admin Dashboard.
Provides background training, job tracking, and model deployment.

Features:
- Trigger manual model retraining
- Track training jobs status
- Integration with ModelRegistry for versioning
- Background job execution
"""
import logging
import threading
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, asdict
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from services.ml.model_registry import ModelRegistry, ModelStatus

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Training job statuses"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TrainingJob:
    """Training job metadata"""
    job_id: str
    model_name: str
    model_type: str
    status: JobStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    triggered_by: str = "system"
    error_message: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        # Convert datetimes to ISO format
        for field in ['created_at', 'started_at', 'completed_at']:
            if result.get(field):
                result[field] = result[field].isoformat()
        return result


class TrainingExecutor:
    """ML Training Executor for Admin Dashboard"""

    def __init__(self):
        """Initialize training executor"""
        self.model_registry = ModelRegistry()
        self._jobs: Dict[str, TrainingJob] = {}
        self._jobs_lock = threading.Lock()
        logger.info("✅ Training Executor initialized")

    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        List all available ML models from registry.

        Returns:
            List of model info dictionaries
        """
        try:
            models = self.model_registry.list_models()

            # Enrich with additional info
            enriched_models = []
            for model in models:
                # Find active training jobs for this model
                active_jobs = [
                    job for job in self._jobs.values()
                    if job.model_name == model['name'] and job.status in [JobStatus.PENDING, JobStatus.RUNNING]
                ]

                enriched_models.append({
                    **model,
                    "has_active_job": len(active_jobs) > 0,
                    "active_job_status": active_jobs[0].status.value if active_jobs else None
                })

            return enriched_models

        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []

    def trigger_training(
        self,
        model_name: str,
        model_type: str = "unknown",
        admin_user: str = "system"
    ) -> Dict[str, Any]:
        """
        Trigger model training job.

        Args:
            model_name: Name of model to train
            model_type: Type of model (volatility, sentiment, risk, etc.)
            admin_user: User who triggered the training

        Returns:
            Job info dictionary
        """
        try:
            # Generate job ID
            job_id = f"{model_name}_{int(time.time())}"

            # Create job
            job = TrainingJob(
                job_id=job_id,
                model_name=model_name,
                model_type=model_type,
                status=JobStatus.PENDING,
                created_at=datetime.utcnow(),
                triggered_by=admin_user
            )

            # Store job
            with self._jobs_lock:
                self._jobs[job_id] = job

            logger.info(f"✅ Training job created: {job_id} for model {model_name} by {admin_user}")

            # Start training in background
            thread = threading.Thread(target=self._run_training_job, args=(job_id,))
            thread.daemon = True
            thread.start()

            return {
                "ok": True,
                "job_id": job_id,
                "model_name": model_name,
                "status": job.status.value,
                "created_at": job.created_at.isoformat()
            }

        except Exception as e:
            logger.error(f"Error triggering training for {model_name}: {e}")
            return {
                "ok": False,
                "error": str(e)
            }

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get training job status.

        Args:
            job_id: Job ID

        Returns:
            Job info dictionary or None if not found
        """
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if job:
                return job.to_dict()
        return None

    def list_jobs(
        self,
        status_filter: Optional[JobStatus] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        List training jobs.

        Args:
            status_filter: Filter by job status (optional)
            limit: Max number of jobs to return

        Returns:
            List of job dictionaries
        """
        with self._jobs_lock:
            jobs = list(self._jobs.values())

        # Filter by status if specified
        if status_filter:
            jobs = [j for j in jobs if j.status == status_filter]

        # Sort by created_at descending (most recent first)
        jobs.sort(key=lambda j: j.created_at, reverse=True)

        # Limit results
        jobs = jobs[:limit]

        return [job.to_dict() for job in jobs]

    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        """
        Cancel a training job.

        Args:
            job_id: Job ID

        Returns:
            Result dictionary
        """
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if not job:
                return {"ok": False, "error": "Job not found"}

            if job.status not in [JobStatus.PENDING, JobStatus.RUNNING]:
                return {"ok": False, "error": f"Cannot cancel job in status {job.status.value}"}

            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.utcnow()

        logger.info(f"✅ Job {job_id} cancelled")
        return {"ok": True, "job_id": job_id, "status": JobStatus.CANCELLED.value}

    def _run_training_job(self, job_id: str):
        """
        Run training job in background (MOCK implementation).

        In production, this would:
        1. Load training data
        2. Train model using appropriate trainer (BTC regime, sentiment, etc.)
        3. Validate model
        4. Register new version in ModelRegistry
        5. Update job status

        For Phase 3, we simulate training with a delay.
        """
        try:
            with self._jobs_lock:
                job = self._jobs[job_id]
                job.status = JobStatus.RUNNING
                job.started_at = datetime.utcnow()

            logger.info(f"🚀 Starting training job {job_id} for {job.model_name}")

            # MOCK: Simulate training (in production, call actual training code)
            # Example: Call scripts/train_models.py functions
            time.sleep(5)  # Simulate training delay

            # MOCK: Simulate success with dummy metrics
            metrics = {
                "accuracy": 0.85 + (hash(job_id) % 100) / 1000,  # Fake metric
                "loss": 0.15 - (hash(job_id) % 100) / 1000,
                "training_time_seconds": 5
            }

            # Update ModelRegistry after training
            try:
                from services.ml.model_registry import ModelStatus

                # Get model manifest (assuming version v1.0 for simplicity)
                manifest = self.model_registry.get_model_manifest(job.model_name, "v1.0")

                # Update metrics in registry
                self.model_registry.update_metrics(
                    job.model_name,
                    "v1.0",
                    validation_metrics=metrics,
                    test_metrics=metrics
                )

                # Update status to TRAINED (will update updated_at automatically)
                self.model_registry.update_status(
                    job.model_name,
                    "v1.0",
                    ModelStatus.TRAINED
                )

                logger.info(f"✅ ModelRegistry updated for {job.model_name}")

            except Exception as e:
                logger.warning(f"⚠️ Failed to update ModelRegistry: {e}")

            with self._jobs_lock:
                job.status = JobStatus.COMPLETED
                job.completed_at = datetime.utcnow()
                job.metrics = metrics

            logger.info(f"✅ Training job {job_id} completed successfully")

        except Exception as e:
            logger.error(f"❌ Training job {job_id} failed: {e}")

            with self._jobs_lock:
                job = self._jobs[job_id]
                job.status = JobStatus.FAILED
                job.completed_at = datetime.utcnow()
                job.error_message = str(e)


# Global training executor instance
training_executor = TrainingExecutor()
