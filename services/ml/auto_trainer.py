"""
ML Auto-Trainer Service

Automatic periodic training of ML models using APScheduler.

Schedule:
- Regime models (stock, btc): Weekly on Sunday at 3am
- Volatility models: Daily at midnight
- Correlation models: Weekly on Sunday at 4am

Features:
- Automatic training based on model age
- Integration with TrainingExecutor
- Admin control endpoints (start/stop/status)
- Error handling and logging
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR

from services.ml.training_executor import training_executor
from services.ml.bourse.training_scheduler import MLTrainingScheduler

logger = logging.getLogger(__name__)


class MLAutoTrainer:
    """
    Automatic ML model training scheduler.

    Runs in background and periodically checks if models need retraining
    based on MLTrainingScheduler rules.
    """

    def __init__(self):
        """Initialize auto-trainer with APScheduler"""
        self.scheduler = BackgroundScheduler(
            timezone='Europe/Zurich',
            job_defaults={
                'coalesce': True,  # Combine multiple missed runs into one
                'max_instances': 1,  # Only one instance of each job at a time
                'misfire_grace_time': 3600  # Allow 1h grace period for missed jobs
            }
        )
        self._is_running = False
        self._last_run: Dict[str, datetime] = {}

        # Add event listeners
        self.scheduler.add_listener(
            self._job_executed_listener,
            EVENT_JOB_EXECUTED | EVENT_JOB_ERROR
        )

        logger.info("✅ ML Auto-Trainer initialized")

    def start(self):
        """Start the auto-trainer scheduler"""
        if self._is_running:
            logger.warning("Auto-trainer already running")
            return

        # Schedule regime models training (weekly - Sunday 3am)
        self.scheduler.add_job(
            func=self._train_regime_models,
            trigger=CronTrigger(day_of_week='sun', hour=3, minute=0),
            id='regime_training_weekly',
            name='Weekly Regime Models Training',
            replace_existing=True
        )

        # Schedule volatility models training (daily - midnight)
        self.scheduler.add_job(
            func=self._train_volatility_models,
            trigger=CronTrigger(hour=0, minute=0),
            id='volatility_training_daily',
            name='Daily Volatility Models Training',
            replace_existing=True
        )

        # Schedule correlation models training (weekly - Sunday 4am)
        self.scheduler.add_job(
            func=self._train_correlation_models,
            trigger=CronTrigger(day_of_week='sun', hour=4, minute=0),
            id='correlation_training_weekly',
            name='Weekly Correlation Models Training',
            replace_existing=True
        )

        self.scheduler.start()
        self._is_running = True

        logger.info("🚀 ML Auto-Trainer started")
        logger.info("   • Regime models: Every Sunday at 3am")
        logger.info("   • Volatility models: Daily at midnight")
        logger.info("   • Correlation models: Every Sunday at 4am")

    def stop(self):
        """Stop the auto-trainer scheduler"""
        if not self._is_running:
            logger.warning("Auto-trainer not running")
            return

        self.scheduler.shutdown(wait=False)
        self._is_running = False
        logger.info("🛑 ML Auto-Trainer stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status and next run times"""
        if not self._is_running:
            return {
                "running": False,
                "jobs": []
            }

        jobs = []
        for job in self.scheduler.get_jobs():
            next_run = job.next_run_time
            jobs.append({
                "id": job.id,
                "name": job.name,
                "next_run": next_run.isoformat() if next_run else None,
                "last_run": self._last_run.get(job.id, None)
            })

        return {
            "running": True,
            "jobs": jobs,
            "timezone": str(self.scheduler.timezone)
        }

    def trigger_now(self, job_id: str) -> Dict[str, Any]:
        """
        Manually trigger a scheduled job immediately.

        Args:
            job_id: Job ID to trigger (e.g., 'regime_training_weekly')

        Returns:
            Result dict with status
        """
        try:
            job = self.scheduler.get_job(job_id)
            if not job:
                return {"ok": False, "error": f"Job {job_id} not found"}

            # Execute job now
            job.modify(next_run_time=datetime.now())

            return {"ok": True, "message": f"Job {job_id} triggered"}
        except Exception as e:
            logger.error(f"Failed to trigger job {job_id}: {e}")
            return {"ok": False, "error": str(e)}

    # ============================================================================
    # Private Methods - Training Logic
    # ============================================================================

    def _train_regime_models(self):
        """
        Check and train regime models (stock_regime_detector, btc_regime_detector, btc_regime_hmm).

        Uses MLTrainingScheduler to determine if retraining is needed.
        """
        logger.info("🔄 Auto-training: Checking regime models...")

        models_to_check = [
            ("stock_regime_detector", "regime"),
            ("btc_regime_detector", "regime"),
            ("btc_regime_hmm", "regime")
        ]

        for model_name, model_type in models_to_check:
            try:
                # Check if model needs retraining
                model_path = self._get_model_path(model_name)

                if MLTrainingScheduler.should_retrain("regime", model_path):
                    logger.info(f"   → Training {model_name} (model is outdated)")

                    result = training_executor.trigger_training(
                        model_name=model_name,
                        model_type=model_type,
                        config=None,  # Use defaults
                        admin_user="auto_trainer"
                    )

                    if result.get("ok"):
                        logger.info(f"   ✅ {model_name} training started (job: {result.get('job_id')})")
                    else:
                        logger.error(f"   ❌ Failed to train {model_name}: {result.get('error')}")
                else:
                    logger.debug(f"   ⏭️  {model_name} is up-to-date, skipping")

            except Exception as e:
                logger.error(f"Error checking {model_name}: {e}", exc_info=True)

        self._last_run['regime_training_weekly'] = datetime.now().isoformat()

    def _train_volatility_models(self):
        """
        Check and train volatility models (volatility_forecaster).

        Uses MLTrainingScheduler to determine if retraining is needed.
        """
        logger.info("🔄 Auto-training: Checking volatility models...")

        models_to_check = [
            ("volatility_forecaster", "volatility")
        ]

        for model_name, model_type in models_to_check:
            try:
                model_path = self._get_model_path(model_name)

                if MLTrainingScheduler.should_retrain("volatility", model_path):
                    logger.info(f"   → Training {model_name} (model is outdated)")

                    result = training_executor.trigger_training(
                        model_name=model_name,
                        model_type=model_type,
                        config=None,  # Use defaults
                        admin_user="auto_trainer"
                    )

                    if result.get("ok"):
                        logger.info(f"   ✅ {model_name} training started (job: {result.get('job_id')})")
                    else:
                        logger.error(f"   ❌ Failed to train {model_name}: {result.get('error')}")
                else:
                    logger.debug(f"   ⏭️  {model_name} is up-to-date, skipping")

            except Exception as e:
                logger.error(f"Error checking {model_name}: {e}", exc_info=True)

        self._last_run['volatility_training_daily'] = datetime.now().isoformat()

    def _train_correlation_models(self):
        """
        Check and train correlation models (correlation_forecaster).

        Uses MLTrainingScheduler to determine if retraining is needed.
        """
        logger.info("🔄 Auto-training: Checking correlation models...")

        # Placeholder: correlation models not implemented yet
        logger.debug("   ⏭️  Correlation models not implemented, skipping")

        self._last_run['correlation_training_weekly'] = datetime.now().isoformat()

    def _get_model_path(self, model_name: str) -> Path:
        """
        Get the file path for a model.

        Args:
            model_name: Name of the model

        Returns:
            Path to the model file
        """
        # Map model names to their file paths
        model_paths = {
            "stock_regime_detector": Path("models/stocks/regime/regime_neural_best.pth"),
            "btc_regime_detector": Path("models/regime/regime_neural_best.pth"),
            "btc_regime_hmm": Path("models/regime/btc_regime_hmm.pkl"),
            "volatility_forecaster": Path("models/volatility/volatility_model_v2.1.pkl")
        }

        return model_paths.get(model_name, Path(f"models/{model_name}.pkl"))

    def _job_executed_listener(self, event):
        """
        Listener for job execution events.

        Logs job execution success/failure.
        """
        job_id = event.job_id

        if event.exception:
            logger.error(f"❌ Job {job_id} failed: {event.exception}")
        else:
            logger.info(f"✅ Job {job_id} completed successfully")


# Global singleton instance
ml_auto_trainer = MLAutoTrainer()
