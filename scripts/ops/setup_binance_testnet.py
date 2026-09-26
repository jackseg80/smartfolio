#!/usr/bin/env python3
"""
Script d'installation et de test pour l'API Binance Testnet

Ce script aide à configurer et tester la connexion à Binance Testnet
pour l'execution engine.
"""

import os
import asyncio
import logging
from typing import Optional
from services.execution.exchange_adapter import ExchangeConfig, ExchangeType, BinanceAdapter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_env_file():
    """Créer un fichier .env avec les paramètres Binance"""
    env_path = '.env'
    
    if os.path.exists(env_path):
        print(f" Le fichier {env_path} existe déjà")
        return
    
    print(" Création du fichier .env...")
    
    with open(env_path, 'w') as f:
        f.write("""# Configuration Crypto Rebalancer - Execution Engine
# Généré automatiquement par setup_binance_testnet.py

# ---- BINANCE TESTNET CONFIG ----
# [Warning]  IMPORTANT: Ces clés sont pour TESTNET uniquement !
# Obtenez vos clés ici: https://testnet.binance.vision/
BINANCE_SANDBOX=true
BINANCE_API_KEY=your_testnet_api_key_here
BINANCE_API_SECRET=your_testnet_api_secret_here

# ---- AUTRES CONFIGS ----
LOG_LEVEL=INFO
""")
    
    print(f"[OK] Fichier {env_path} créé")
    print(" Maintenant, éditez ce fichier et ajoutez vos vraies clés API Binance Testnet")
    print(" Guide: https://testnet.binance.vision/")

async def test_binance_connection(api_key: str, api_secret: str) -> bool:
    """Tester la connexion à Binance"""
    print(f" Test de connexion Binance avec clé {api_key[:8]}...")
    
    config = ExchangeConfig(
        name="binance_test",
        type=ExchangeType.CEX,
        api_key=api_key,
        api_secret=api_secret,
        sandbox=True,  # Toujours testnet pour ce script
        fee_rate=0.001,
        min_order_size=10.0
    )
    
    adapter = BinanceAdapter(config)
    
    try:
        # Test de connexion
        connected = await adapter.connect()
        
        if not connected:
            print("[Error] Échec de la connexion")
            return False
        
        print("[OK] Connexion réussie!")
        
        # Test de récupération d'informations
        print(" Test des fonctionnalités...")
        
        # Test balance
        try:
            balance = await adapter.get_balance('USDT')
            print(f" Balance USDT: {balance}")
        except Exception as e:
            print(f"[Warning]  Erreur balance: {e}")
        
        # Test prix
        try:
            price = await adapter.get_current_price('BTC/USDT')
            print(f" Prix BTC/USDT: ${price}")
        except Exception as e:
            print(f"[Warning]  Erreur prix: {e}")
        
        # Test paires de trading
        try:
            pairs = await adapter.get_trading_pairs()
            print(f" Paires disponibles: {len(pairs)} (exemples: {', '.join([p.symbol for p in pairs[:5]])})")
        except Exception as e:
            print(f"[Warning]  Erreur paires: {e}")
        
        await adapter.disconnect()
        return True
        
    except Exception as e:
        print(f"[Error] Erreur lors du test: {e}")
        return False

async def interactive_test():
    """Test interactif avec saisie des clés"""
    print(" Test interactif de l'API Binance Testnet")
    print()
    print(" Pour commencer, vous avez besoin de clés API Testnet:")
    print("1. Allez sur https://testnet.binance.vision/")  
    print("2. Créez un compte ou connectez-vous")
    print("3. Générez vos clés API")
    print("4. Copiez API Key et Secret Key")
    print()
    
    api_key = input(" Entrez votre API Key Testnet: ").strip()
    api_secret = input(" Entrez votre API Secret Testnet: ").strip()
    
    if not api_key or not api_secret:
        print("[Error] Clés API manquantes")
        return
    
    success = await test_binance_connection(api_key, api_secret)
    
    if success:
        print("\n[OK] Test réussi ! Votre configuration Binance fonctionne.")
        print(" Vous pouvez maintenant utiliser l'execution engine avec Binance.")
    else:
        print("\n[Error] Test échoué. Vérifiez vos clés API.")

def check_dependencies():
    """Vérifier les dépendances Python"""
    print(" Vérification des dépendances...")
    
    missing = []
    
    try:
        import binance
        print("[OK] python-binance installé")
    except ImportError:
        missing.append("python-binance")
    
    try:
        import dotenv
        print("[OK] python-dotenv installé")
    except ImportError:
        missing.append("python-dotenv")
    
    if missing:
        print(f"[Error] Dépendances manquantes: {', '.join(missing)}")
        print(" Pour installer:")
        print(f"   pip install {' '.join(missing)}")
        return False
    
    print("[OK] Toutes les dépendances sont présentes")
    return True

async def main():
    """Function principale"""
    print(" Setup Binance Testnet - Crypto Rebalancer")
    print("=" * 50)
    
    # 1. Vérifier dépendances
    if not check_dependencies():
        return
    
    print()
    
    # 2. Charger variables d'environnement si .env existe
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print(" Variables d'environnement chargées")
    except ImportError:
        print("[Warning]  python-dotenv non installé, variables d'environnement système utilisées")
    
    # 3. Vérifier si les clés sont configurées
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    
    if api_key and api_secret and api_key != 'your_testnet_api_key_here':
        print(f" Clés Binance trouvées dans l'environnement")
        success = await test_binance_connection(api_key, api_secret)
        if success:
            print("\n Configuration Binance opérationnelle!")
        else:
            print("\n[Error] Problème avec la configuration existante")
    else:
        print(" Pas de clés Binance configurées")
        
        # 4. Créer .env si nécessaire
        if not os.path.exists('.env'):
            create_env_file()
        
        # 5. Test interactif
        print("\n" + "=" * 30)
        choice = input("Voulez-vous tester avec des clés maintenant? (y/N): ").strip().lower()
        
        if choice == 'y':
            await interactive_test()
        else:
            print(" Éditez le fichier .env avec vos clés et relancez ce script")

if __name__ == "__main__":
    asyncio.run(main())