#!/usr/bin/env python3
"""
Test E2E du système d'exécution

Ce script teste le workflow complet:
1. Génération d'un plan de rebalancement
2. Validation du plan d'exécution  
3. Exécution en mode dry-run
4. Monitoring du progrès
"""

import requests
import time
import pytest
from tests.e2e.conftest import requires_server

API_BASE = "http://localhost:8001"

@requires_server
def test_execution_workflow():
    """Test du workflow complet d'exécution"""
    
    print("[TEST] E2E du systeme d'execution")
    print("=" * 50)
    
    # 1. Générer un plan de rebalancement
    print("\n1. Génération du plan de rebalancement...")
    
    rebalance_payload = {
        "primary_symbols": {
            "BTC": ["BTC", "TBTC", "WBTC"],
            "ETH": ["ETH", "WSTETH", "STETH", "RETH", "WETH"],
            "SOL": ["SOL", "JUPSOL", "JITOSOL"]
        },
        "sub_allocation": "proportional",
        "min_trade_usd": 25,
        "dynamic_targets_pct": {
            "BTC": 40,
            "ETH": 25, 
            "Stablecoins": 15,
            "SOL": 10,
            "L1/L0 majors": 10
        }
    }
    
    response = requests.post(
        f"{API_BASE}/rebalance/plan?source=coingecko&min_usd=1&pricing=auto&dynamic_targets=true",
        json=rebalance_payload
    )
    
    if response.status_code != 200:
        print(f"[Error] Erreur génération plan: {response.status_code}")
        print(response.text)
        return False
    
    rebalance_plan = response.json()
    actions = rebalance_plan.get("actions", [])
    
    print(f"[OK] Plan généré: {len(actions)} actions, ${rebalance_plan.get('total_usd', 0):,.2f} total")
    
    # 2. Valider le plan d'exécution
    print("\n2. Validation du plan d'exécution...")
    
    validation_payload = {
        "rebalance_actions": actions,
        "metadata": {
            "dynamic_targets_used": True,
            "ccs_score": 45,
            "source_plan": rebalance_plan
        },
        "dry_run": True,
        "max_parallel": 3
    }
    
    response = requests.post(
        f"{API_BASE}/execution/validate-plan",
        json=validation_payload
    )
    
    if response.status_code != 200:
        print(f"[Error] Erreur validation: {response.status_code}")
        print(response.text)
        return False
    
    validation = response.json()
    plan_id = validation["plan_id"]
    
    print(f"[OK] Plan validé: {validation['total_orders']} ordres, "
          f"${validation['total_volume']:,.2f} volume")
    
    if validation["errors"]:
        print(f"[Warning] Erreurs: {validation['errors']}")
        return False
    
    if validation["warnings"]:
        print(f"[Warning] Avertissements: {validation['warnings']}")
    
    # 3. Connecter les exchanges
    print("\n3. Connexion aux exchanges...")
    
    response = requests.post(f"{API_BASE}/execution/exchanges/connect")
    if response.status_code == 200:
        connect_result = response.json()
        print(f"[OK] Exchanges connectés: {connect_result['message']}")
    else:
        print(f"[Warning] Problème connexion exchanges: {response.status_code}")
    
    # 4. Lancer l'exécution
    print("\n4. Lancement de l'exécution (dry-run)...")
    
    response = requests.post(
        f"{API_BASE}/execution/execute-plan?plan_id={plan_id}&dry_run=true&max_parallel=2"
    )
    
    if response.status_code != 200:
        print(f"[Error] Erreur lancement exécution: {response.status_code}")
        print(response.text)
        return False
    
    execution = response.json()
    print(f"[OK] Exécution lancée: {execution['message']}")
    print(f"[Pending] Durée estimée: {execution.get('estimated_duration_seconds', 0):.1f}s")
    
    # 5. Monitoring du progrès
    print("\n5. Monitoring du progrès...")
    
    max_wait = 60  # 60 secondes max
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        response = requests.get(f"{API_BASE}/execution/status/{plan_id}")
        
        if response.status_code == 200:
            status = response.json()
            
            print(f" Progrès: {status['completion_percentage']:.1f}% - "
                  f"{status['completed_orders']}/{status['total_orders']} ordres - "
                  f"Succès: {status['success_rate']:.1f}%")
            
            if not status['is_active']:
                print(f" Exécution terminée: {status['status']}")
                break
        else:
            print(f"[Warning] Erreur statut: {response.status_code}")
        
        time.sleep(2)
    
    # 6. Résultats finaux
    print("\n6. Résultats finaux...")
    
    response = requests.get(f"{API_BASE}/execution/orders/{plan_id}")
    if response.status_code == 200:
        orders_detail = response.json()
        orders = orders_detail["orders"]
        
        completed = sum(1 for o in orders if o["status"] == "filled")
        failed = sum(1 for o in orders if o["status"] == "failed")
        total_fees = sum(o["fees"] for o in orders)
        
        print(" Résultats détaillés:")
        print(f"   [OK] Complétés: {completed}/{len(orders)}")
        print(f"   [Error] Échecs: {failed}/{len(orders)}")
        print(f"    Frais totaux: ${total_fees:.4f}")
        
        # Afficher quelques ordres exemple
        print("\n Exemples d'ordres:")
        for i, order in enumerate(orders[:3]):
            status_emoji = "OK" if order["status"] == "filled" else "Error" if order["status"] == "failed" else "Pending"
            avg_price = order.get('avg_fill_price') or 0
            print(f"   {status_emoji} {order['alias']}: {order['action']} ${abs(order['usd_amount']):.2f} "
                  f"@ ${avg_price:.2f}")
    
    # 7. Statut global du pipeline
    print("\n7. Statut du pipeline...")
    
    response = requests.get(f"{API_BASE}/execution/pipeline-status")
    if response.status_code == 200:
        pipeline = response.json()
        print(f" Pipeline: {pipeline['pipeline_status']}")
        print(f" Statistiques: {pipeline['statistics']['total_plans']} plans, "
              f"{pipeline['statistics']['success_rate']:.1f}% succès")
    
    print("\n Test E2E terminé avec succès!")
    return True

if __name__ == "__main__":
    try:
        success = test_execution_workflow()
        if success:
            print("\n[OK] Tous les tests sont passés!")
        else:
            print("\n[Error] Certains tests ont échoué!")
    except Exception as e:
        print(f"\n Erreur durant les tests: {e}")
        import traceback
        traceback.print_exc()