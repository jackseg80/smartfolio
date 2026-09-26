#!/usr/bin/env python3
"""
Test de l'interface optimisée avec section pliable et boutons supprimés
"""

import requests

def test_optimized_interface():
    """Test l'interface optimisée"""
    print("Test de l'interface optimisée")
    print("=" * 50)
    
    BASE_URL = "http://localhost:8080"
    
    # Test 1: Vérifier rebalance.html optimisé
    print("\n1. Test rebalance.html optimisé...")
    try:
        resp = requests.get(f"{BASE_URL}/static/rebalance.html", timeout=10)
        if resp.status_code == 200:
            content = resp.text
            
            # Vérifications des améliorations
            has_collapsible_section = "onclick=\"toggleStrategiesSection()\"" in content
            has_toggle_function = "function toggleStrategiesSection()" in content
            no_generate_plan_button = "Générer le plan" not in content
            has_sync_ccs_button = " Sync CCS" in content
            has_placeholder_handling = "_isPlaceholder" in content
            has_error_handling = "_isError" in content
            has_localStorage_persistence = "localStorage.setItem('strategies_section_collapsed'" in content
            
            print(f"   Section pliable: {'OUI' if has_collapsible_section else 'NON'}")
            print(f"   Fonction toggle: {'OUI' if has_toggle_function else 'NON'}")
            print(f"   Bouton 'Générer le plan' supprimé: {'OUI' if no_generate_plan_button else 'NON'}")
            print(f"   Bouton 'Sync CCS': {'OUI' if has_sync_ccs_button else 'NON'}")
            print(f"   Gestion placeholder: {'OUI' if has_placeholder_handling else 'NON'}")
            print(f"   Gestion d'erreur: {'OUI' if has_error_handling else 'NON'}")
            print(f"   Persistence état: {'OUI' if has_localStorage_persistence else 'NON'}")
            
            if all([has_collapsible_section, has_toggle_function, no_generate_plan_button, 
                   has_sync_ccs_button, has_placeholder_handling, has_error_handling, 
                   has_localStorage_persistence]):
                print("   [OK] rebalance.html correctement optimisé")
            else:
                print("   [Warning] Certaines optimisations manquent")
                
        else:
            print(f"   [Error] Impossible d'accéder à rebalance.html (HTTP {resp.status_code})")
            
    except Exception as e:
        print(f"   [Error] Erreur: {e}")
    
    # Test 2: Vérifier risk-dashboard.html nettoyé
    print("\n2. Test risk-dashboard.html nettoyé...")
    try:
        resp = requests.get(f"{BASE_URL}/static/risk-dashboard.html", timeout=10)
        if resp.status_code == 200:
            content = resp.text
            
            has_info_message = "Nouvelle méthode d'application" in content
            no_apply_targets_button = "Apply Targets" not in content or content.count("Apply Targets") == 0
            no_apply_targets_function = "window.applyTargetsAction" not in content
            has_sync_instruction = " Sync CCS" in content
            
            print(f"   Message informatif: {'OUI' if has_info_message else 'NON'}")
            print(f"   Bouton 'Apply Targets' supprimé: {'OUI' if no_apply_targets_button else 'NON'}")
            print(f"   Fonction applyTargetsAction supprimée: {'OUI' if no_apply_targets_function else 'NON'}")
            print(f"   Instructions Sync CCS: {'OUI' if has_sync_instruction else 'NON'}")
            
            if all([has_info_message, no_apply_targets_button, no_apply_targets_function, has_sync_instruction]):
                print("   [OK] risk-dashboard.html correctement nettoyé")
            else:
                print("   [Warning] Nettoyage incomplet")
                
        else:
            print(f"   [Error] Impossible d'accéder à risk-dashboard.html (HTTP {resp.status_code})")
            
    except Exception as e:
        print(f"   [Error] Erreur: {e}")
    
    print("\n" + "=" * 50)
    print("NOUVELLES FONCTIONNALITÉS À TESTER MANUELLEMENT")
    print("=" * 50)
    print("""
 WORKFLOW OPTIMISÉ À TESTER:

1. INTERFACE PLIABLE:
   - Aller sur rebalance.html
   - Cliquer sur le titre " Stratégies Prédéfinies"
   - Vérifier que la section se plie/déplie
   - Rafraîchir la page et vérifier que l'état est sauvegardé

2. STRATÉGIE DYNAMIQUE:
   - Si aucune donnée CCS: voir stratégie placeholder "En attente de synchronisation"
   - Cliquer " Sync CCS" sans données: voir "Aucune donnée CCS récente trouvée"
   - Avoir des données CCS puis Sync: voir stratégie "Strategic (Dynamic)" fonctionnelle

3. WORKFLOW SIMPLIFIÉ:
   - Plus de bouton "Générer le plan" (redondant)
   - Sélectionner une stratégie → Appliquer → Plan généré automatiquement
   - Plus de confusion avec l'ancien "Apply Targets" dans risk-dashboard

4. GESTION D'ERREURS:
   - Si erreur sync CCS: stratégie affichée avec icône [Warning] et non-cliquable
   - Autres stratégies continuent de fonctionner normalement

AVANTAGES DE L'OPTIMISATION:
[OK] Interface plus épurée
[OK] Section pliable pour gagner de la place
[OK] Workflow simplifié et intuitif
[OK] Gestion robuste des erreurs
[OK] Plus de doublons fonctionnels
[OK] État de l'interface persisté
""")

if __name__ == "__main__":
    test_optimized_interface()
