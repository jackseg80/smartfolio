"""Package verified ML code/data only. Never includes user data or credentials."""
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from services.ml.reliability import CapabilityService, code_version, PROTOCOL_ID

BASE = "806338b522e89647c3f49260e0f418dbea1da761"
NEW_FILES = (
    'docs/audit/MARKET_OPPORTUNITIES_DEPLOYMENT_2026-10-02.md',
    'docs/audit/MARKET_OPPORTUNITIES_UI_2026-10-02.md',
    'deploy/ml-production/release_robot2.py',
    'docs/audit/MARKET_OPPORTUNITIES_RELAY_2026-10-02.md',
    'docs/audit/MARKET_OPPORTUNITIES_AUDIT_2026-10-01.md',
    'tests/unit/test_market_opportunities_api.py',
    'static/tests/market-opportunities.test.js',
    'services/ml/bourse/fund_exposure.py',
    'services/ml/bourse/market_analysis.py',
    'services/ml/bourse/market_snapshot.py',
    'static/components/market-opportunities.js',
    'tests/unit/test_market_opportunities_v2.py',

    "static/components/stock-ml-insights.js", "static/tests/stock-ml-insights.test.js", "tests/unit/test_stock_ml_loading.py",
    "services/ml/live_observations.py", "scripts/acquire_cycle_reference.py", "tests/unit/test_ml_oct1_repairs.py",
    "scripts/repair_ml_stock_observations.py", "static/components/ml-result-table.js",
    "static/core/risk-request.js", "static/core/role-policy.js",
    "static/tests/ml-feedback-regressions.test.js", "tests/unit/test_ml_feedback_regressions.py",
    "static/tests/bourse-source-context.test.js",
    "docs/audit/ML_RELIABILITY_REPORTED_REGRESSIONS_2026-09-30.md",
    "services/ml/cycle_diagnostics.py", "tests/unit/test_ml_reported_regressions.py",
    "config/ml_capability_registry.json",
    "services/ml/reliability.py", "services/ml/risk_evaluation.py",
    "services/ml/portfolio_context.py", "scripts/evaluate_ml_risk.py",
    "scripts/package_ml_reliability.py", "static/components/ml-overview.js",
    "static/core/selected-source.js", "static/tests/selected-source.test.js", "static/tests/ml-overview.test.js", "tests/unit/test_ml_reliability.py", "tests/unit/test_ml_preview_regressions.py",
    "docs/audit/ML_RELIABILITY_PROTOCOL_2026-09-30.md",
    "docs/audit/ML_RELIABILITY_INVENTORY_2026-09-30.md",
    "docs/audit/ML_RELIABILITY_DELIVERY_2026-09-30.md",
    "docs/audit/ML_RELIABILITY_RELEASE_2026-09-30.md", "docs/audit/ML_RELIABILITY_ROBOT2_PREVIEW_2026-09-30.md",
)
SKIP = {"config/score_registry.json", "data/taxonomy_aliases.json"}
def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])
def sha(content):
    return hashlib.sha256(content).hexdigest()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/ml-reliability/ml-reliability-release.zip")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((ROOT/"outputs/ml-reliability").resolve()):
        raise ValueError("Release output must stay in the isolated ML output directory")
    if git("merge-base", BASE, "HEAD").decode().strip() != BASE:
        raise ValueError("Review the production base before packaging")
    service = CapabilityService(refresh_observations=False)
    registry = json.loads((ROOT/"models/validated_risk/registry.json").read_text(encoding="utf-8"))
    for name, entry in registry.items():
        market, remainder = name.removesuffix(".json").split("_", 1)
        asset, days = remainder.rsplit("_", 1)
        artifact, _ = service.artifact(market, asset, int(days.removesuffix("d")))
        _, receipt = service.history(market, asset)
        if artifact["dataset_id"] != receipt["dataset_id"] or entry["governance_eligible"] is not False:
            raise ValueError("Artifact/dataset mismatch or unexpected governance eligibility")
    paths = {}
    for name in git("diff", "--name-only", "--diff-filter=M", BASE).decode().splitlines():
        if name in SKIP:
            continue
        content = (ROOT/name).read_bytes()
        original = git("show", BASE+":"+name)
        if content.replace(b"\r\n",b"\n") == original.replace(b"\r\n",b"\n"):
            continue
        if not name.startswith(("api/", "services/", "static/", "tests/", "docs/")) and name not in (".gitignore", ".gitattributes", "requirements.txt"):
            raise ValueError("Unexpected modified path: "+name)
        paths["code/"+name] = ROOT/name
    for name in NEW_FILES:
        if not (ROOT/name).is_file():
            raise ValueError("Required delivery file is missing: "+name)
        paths["code/"+name] = ROOT/name
    for directory, suffixes in (("data/ml_verified", {".json",".csv"}), ("models/validated_risk", {".json"}), ("outputs/ml-reliability/evaluations", {".json"})):
        for path in sorted((ROOT/directory).rglob("*")):
            if path.is_file() and path.suffix in suffixes:
                paths["public/"+str(path.relative_to(ROOT)).replace("\\","/")] = path
    for name in ("checks-summary.json", "production-readonly.json"):
        path = ROOT/"outputs/ml-reliability"/name
        if path.exists():
            paths["proof/"+name] = path
    manifest = dict(base_commit=BASE, code_version=code_version(), protocol_id=PROTOCOL_ID,
        created_at=datetime.now(timezone.utc).isoformat(), governance_integration=False,
        private_data_included=False, entries=[
            dict(path=name, sha256=sha(path.read_bytes()), bytes=path.stat().st_size)
            for name,path in sorted(paths.items())])
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output,"w",ZIP_DEFLATED) as bundle:
        for name,path in sorted(paths.items()):
            bundle.write(path,name)
        bundle.writestr("manifest.json",json.dumps(manifest,indent=2))
    with ZipFile(output) as bundle:
        for row in manifest["entries"]:
            if sha(bundle.read(row["path"])) != row["sha256"]:
                raise ValueError("Packaged file hash mismatch")
        if any("private/" in name or "data/users/" in name or ".env" in name for name in bundle.namelist()):
            raise ValueError("Private content was included")
    (output.with_suffix(".manifest.json")).write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    (output.with_suffix(".sha256")).write_text(sha(output.read_bytes())+"  "+output.name+"\n",encoding="utf-8")
    print(json.dumps(dict(package=str(output),files=len(paths),artifacts=len(registry),sha256=sha(output.read_bytes()))))
if __name__ == "__main__":
    main()

