#!/usr/bin/env python3
"""
Test final des corrections
"""

import requests

def test_fixes():
    """Test que les corrections sont appliquées"""
    print("Test des corrections appliquées")
    print("=" * 40)
    
    BASE_URL = "http://localhost:8080"
    
    # Test 1: Vérifier que rebalance.html a les logs ajoutés
    print("\n1. Vérification des logs ajoutés...")
    try:
        resp = requests.get(f"{BASE_URL}/static/rebalance.html", timeout=10)
        if resp.status_code == 200:
            content = resp.text
            
            has_debug_logs = "console.log(' checkForNewTargets called')" in content
            has_ccs_extraction = "const ccsMatch = targetsData.strategy.match(/(\\d+)/)" in content
            has_apply_logs = "console.log(` Applying CCS targets:" in content
            
            print(f"   Logs de debug: {'OK' if has_debug_logs else 'Error'}")
            print(f"   Extraction CCS améliorée: {'OK' if has_ccs_extraction else 'Error'}")
            print(f"   Logs d'application: {'OK' if has_apply_logs else 'Error'}")
            
            if has_debug_logs and has_ccs_extraction and has_apply_logs:
                print("   [OK] Corrections appliquées dans rebalance.html")
            else:
                print("   [Error] Certaines corrections manquent")
                
        else:
            print(f"   [Error] Impossible d'accéder à rebalance.html (HTTP {resp.status_code})")
            
    except Exception as e:
        print(f"   [Error] Erreur: {e}")
    
    # Test 2: Vérifier les corrections dans targets-coordinator
    print("\n2. Vérification des corrections dans targets-coordinator...")
    try:
        resp = requests.get(f"{BASE_URL}/static/modules/targets-coordinator.js", timeout=10)
        if resp.status_code == 200:
            content = resp.text
            
            has_ccs_fallback = "strategy = 'Macro (CCS unavailable)'" in content
            has_cycle_fallback = "strategy = 'Macro (Cycle unavailable)'" in content
            has_localstorage_save = "localStorage.setItem('last_targets'" in content
            
            print(f"   Fallback CCS: {'OK' if has_ccs_fallback else 'Error'}")
            print(f"   Fallback Cycle: {'OK' if has_cycle_fallback else 'Error'}")
            print(f"   Sauvegarde localStorage: {'OK' if has_localstorage_save else 'Error'}")
            
            if has_ccs_fallback and has_cycle_fallback and has_localstorage_save:
                print("   [OK] Corrections appliquées dans targets-coordinator.js")
            else:
                print("   [Error] Certaines corrections manquent")
                
        else:
            print(f"   [Error] Impossible d'accéder à targets-coordinator.js (HTTP {resp.status_code})")
            
    except Exception as e:
        print(f"   [Error] Erreur: {e}")
    
    # Test 3: Vérifier les corrections dans risk-dashboard
    print("\n3. Vérification des corrections dans risk-dashboard...")
    try:
        resp = requests.get(f"{BASE_URL}/static/risk-dashboard.html", timeout=10)
        if resp.status_code == 200:
            content = resp.text
            
            has_unconditional_targets = "if (activeTab === 'targets') {" in content
            has_strategy_functions = "window.applyStrategy = async function" in content
            
            print(f"   Rendu inconditionnel des targets: {'OK' if has_unconditional_targets else 'Error'}")
            print(f"   Fonctions de stratégie: {'OK' if has_strategy_functions else 'Error'}")
            
            if has_unconditional_targets and has_strategy_functions:
                print("   [OK] Corrections appliquées dans risk-dashboard.html")
            else:
                print("   [Error] Certaines corrections manquent")
                
        else:
            print(f"   [Error] Impossible d'accéder à risk-dashboard.html (HTTP {resp.status_code})")
            
    except Exception as e:
        print(f"   [Error] Erreur: {e}")
    
    print("\n" + "=" * 40)
    print("INSTRUCTIONS DE TEST MANUEL")
    print("=" * 40)
    print("""
 POUR TESTER LES CORRECTIONS:

1. OUVRIR REBALANCE PAGE:
   http://localhost:8080/static/rebalance.html
   
2. OUVRIR DEVTOOLS (F12) -> CONSOLE:
   - Vous devriez voir des messages " checkForNewTargets called" toutes les 2 secondes
   
3. TESTER LA COMMUNICATION:
   - Coller ce code dans la console:
   
   const testData = {
       targets: { "BTC": 40, "ETH": 30, "Stablecoins": 20, "Others": 10 },
       timestamp: new Date().toISOString(),
       strategy: "Test Manual CCS 75",
       source: "risk-dashboard-ccs"
   };
   localStorage.setItem('last_targets', JSON.stringify(testData));
   
4. OBSERVER LES LOGS:
   - Vous devriez voir " Applying CCS targets: Test Manual CCS 75 (CCS: 75)"
   - L'indicateur " CCS 75" devrait apparaître
   - Un plan devrait être généré automatiquement
   
5. TESTER RISK DASHBOARD:
   - Ouvrir: http://localhost:8080/static/risk-dashboard.html
   - Cliquer sur "Strategic Targets"
   - Tester les boutons de stratégie (devraient fonctionner même sans CCS)
   - Cliquer "Apply Targets" (devrait sauvegarder dans localStorage)

RÉSULTATS ATTENDUS:
[OK] Logs de debug apparaissent dans la console
[OK] Boutons de stratégie fonctionnent
[OK] Apply Targets sauvegarde les données
[OK] Rebalance détecte automatiquement les nouveaux targets
[OK] Indicateur "Targets dynamiques" apparaît
[OK] Plan généré utilise les allocations CCS
""")

if __name__ == "__main__":
    test_fixes()
