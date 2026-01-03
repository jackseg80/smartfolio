"""
Script pour générer et configurer les passwords des utilisateurs.

Usage:
    # Générer des passwords aléatoires pour tous les users sans password
    python scripts/setup_passwords.py

    # Définir un password spécifique pour un utilisateur
    python scripts/setup_passwords.py --user jack --password "mon_super_password"

    # Regénérer tous les passwords (force)
    python scripts/setup_passwords.py --force
"""
import sys
import os
import json
import secrets
import string
from pathlib import Path
import bcrypt

# Ajouter le répertoire parent au path pour pouvoir importer api
sys.path.insert(0, str(Path(__file__).parent.parent))

# Paths
USERS_CONFIG_PATH = Path(__file__).parent.parent / "config" / "users.json"


def generate_secure_password(length: int = 16) -> str:
    """
    Génère un password sécurisé aléatoire.

    Args:
        length: Longueur du password (default: 16)

    Returns:
        str: Password aléatoire avec lettres, chiffres et caractères spéciaux
    """
    # Caractères autorisés (sans caractères ambigus comme 0, O, l, I)
    alphabet = string.ascii_letters + string.digits + "!@#$%&*-_+=?"
    alphabet = alphabet.replace('0', '').replace('O', '').replace('l', '').replace('I', '')

    # Générer password sécurisé
    password = ''.join(secrets.choice(alphabet) for _ in range(length))

    # Assurer au moins un de chaque type
    if not any(c.islower() for c in password):
        password = secrets.choice(string.ascii_lowercase) + password[1:]
    if not any(c.isupper() for c in password):
        password = secrets.choice(string.ascii_uppercase) + password[1:]
    if not any(c.isdigit() for c in password):
        password = secrets.choice(string.digits) + password[1:]

    return password


def hash_password(password: str) -> str:
    """
    Hash un password avec bcrypt.

    Args:
        password: Password en clair

    Returns:
        str: Hash bcrypt (format string UTF-8)
    """
    # Convertir le password en bytes
    password_bytes = password.encode('utf-8')

    # Générer le salt et hasher
    salt = bcrypt.gensalt(rounds=12)  # Cost factor 12 (sécurité standard)
    hashed = bcrypt.hashpw(password_bytes, salt)

    # Retourner le hash en string UTF-8
    return hashed.decode('utf-8')


def load_users_config() -> dict:
    """
    Charge la configuration des utilisateurs.

    Returns:
        dict: Configuration users.json
    """
    if not USERS_CONFIG_PATH.exists():
        print(f"❌ Error: Users config not found at {USERS_CONFIG_PATH}")
        sys.exit(1)

    with open(USERS_CONFIG_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_users_config(config: dict):
    """
    Sauvegarde la configuration des utilisateurs.

    Args:
        config: Configuration à sauvegarder
    """
    with open(USERS_CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"✅ Users config saved to {USERS_CONFIG_PATH}")


def setup_user_password(user_id: str, password: str = None, force: bool = False) -> tuple[str, str]:
    """
    Configure le password d'un utilisateur.

    Args:
        user_id: ID de l'utilisateur
        password: Password en clair (généré si None)
        force: Forcer même si password_hash existe déjà

    Returns:
        tuple[str, str]: (password_plain, password_hash)
    """
    config = load_users_config()

    # Trouver l'utilisateur
    user = None
    for u in config.get("users", []):
        if u.get("id") == user_id:
            user = u
            break

    if not user:
        print(f"❌ Error: User '{user_id}' not found in config")
        sys.exit(1)

    # Vérifier si password existe déjà
    if user.get("password_hash") and not force:
        print(f"⚠️  User '{user_id}' already has a password. Use --force to override.")
        return None, user.get("password_hash")

    # Générer ou utiliser le password fourni
    if password is None:
        password = generate_secure_password()
        print(f"🔑 Generated secure password for '{user_id}': {password}")
    else:
        print(f"🔑 Using provided password for '{user_id}'")

    # Hasher le password
    password_hash = hash_password(password)

    # Mettre à jour la config
    user["password_hash"] = password_hash

    # Sauvegarder
    save_users_config(config)

    return password, password_hash


def setup_all_passwords(force: bool = False):
    """
    Configure les passwords pour tous les utilisateurs sans password_hash.

    Args:
        force: Forcer la regénération même si password_hash existe
    """
    config = load_users_config()
    users = config.get("users", [])

    if not users:
        print("❌ No users found in config")
        return

    print(f"\n{'='*60}")
    print(f"SmartFolio - Password Setup")
    print(f"{'='*60}\n")

    results = []

    for user in users:
        user_id = user.get("id")
        has_password = bool(user.get("password_hash"))

        if has_password and not force:
            print(f"⏭️  Skipping '{user_id}' (already has password)")
            continue

        # Générer password
        password = generate_secure_password()
        password_hash = hash_password(password)

        # Mettre à jour
        user["password_hash"] = password_hash

        results.append({
            "user_id": user_id,
            "label": user.get("label", user_id),
            "password": password,
            "roles": user.get("roles", [])
        })

        action = "regenerated" if has_password else "generated"
        print(f"✅ Password {action} for '{user_id}' ({user.get('label', user_id)})")

    # Sauvegarder
    if results:
        save_users_config(config)

        # Afficher résumé
        print(f"\n{'='*60}")
        print(f"Password Summary - SAVE THESE CREDENTIALS SECURELY")
        print(f"{'='*60}\n")

        for result in results:
            print(f"User: {result['label']} ({result['user_id']})")
            print(f"Password: {result['password']}")
            print(f"Roles: {', '.join(result['roles'])}")
            print("-" * 60)

        print(f"\n⚠️  WARNING: Save these passwords now! They cannot be retrieved later.")
        print(f"✅ Setup complete. {len(results)} password(s) configured.\n")
    else:
        print(f"\n✅ No passwords needed to be updated.\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Setup passwords for SmartFolio users"
    )
    parser.add_argument(
        "--user",
        type=str,
        help="User ID to setup password for (e.g., 'jack')"
    )
    parser.add_argument(
        "--password",
        type=str,
        help="Specific password to set (generates random if not provided)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force password regeneration even if already exists"
    )

    args = parser.parse_args()

    if args.user:
        # Setup password for specific user
        setup_user_password(args.user, args.password, args.force)
    else:
        # Setup passwords for all users
        if args.password:
            print("❌ Error: --password can only be used with --user")
            sys.exit(1)
        setup_all_passwords(args.force)


if __name__ == "__main__":
    main()
