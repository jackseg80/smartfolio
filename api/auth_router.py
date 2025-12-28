"""
Router d'authentification avec JWT.
Login/Logout endpoints + utilitaires password hashing.
"""
from __future__ import annotations
from datetime import datetime, timedelta
from typing import Optional
import logging
import os

from fastapi import APIRouter, HTTPException, status, Form, Header
import bcrypt
from jose import JWTError, jwt

from api.config.users import get_user_info, is_allowed_user
from api.utils import success_response, error_response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])

# Configuration JWT
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production-please")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 7  # Token valide 7 jours


# ============================================================================
# Password Hashing Utilities
# ============================================================================

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Vérifie si un password en clair correspond au hash bcrypt.

    Args:
        plain_password: Password en clair
        hashed_password: Hash bcrypt stocké (string)

    Returns:
        bool: True si le password correspond
    """
    try:
        # Convertir en bytes
        password_bytes = plain_password.encode('utf-8')
        hash_bytes = hashed_password.encode('utf-8')

        # Vérifier avec bcrypt
        return bcrypt.checkpw(password_bytes, hash_bytes)
    except Exception as e:
        logger.error(f"Password verification error: {e}")
        return False


def get_password_hash(password: str) -> str:
    """
    Génère un hash bcrypt pour un password.

    Args:
        password: Password en clair

    Returns:
        str: Hash bcrypt (string UTF-8)
    """
    # Convertir en bytes
    password_bytes = password.encode('utf-8')

    # Générer salt et hasher
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(password_bytes, salt)

    # Retourner en string
    return hashed.decode('utf-8')


# ============================================================================
# JWT Token Utilities
# ============================================================================

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Crée un JWT token avec expiration.

    Args:
        data: Payload du token (doit contenir "sub" avec user_id)
        expires_delta: Durée de validité (default: 7 jours)

    Returns:
        str: JWT token encodé
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)

    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow()
    })

    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> Optional[dict]:
    """
    Décode et valide un JWT token.

    Args:
        token: JWT token à décoder

    Returns:
        dict: Payload du token si valide, None sinon
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError as e:
        logger.debug(f"JWT decode error: {e}")
        return None


# ============================================================================
# Authentication Endpoints
# ============================================================================

@router.post("/login")
async def login(
    username: str = Form(...),
    password: str = Form(...)
):
    """
    Endpoint de login avec username/password.

    Retourne un JWT token valide 7 jours si credentials corrects.

    Args:
        username: User ID (form data)
        password: Password en clair (form data)

    Returns:
        {
            "ok": true,
            "data": {
                "token": "eyJhbGc...",
                "token_type": "bearer",
                "expires_in": 604800,
                "user": {
                    "id": "jack",
                    "label": "Jack",
                    "roles": ["admin"]
                }
            }
        }

    Raises:
        HTTPException: 401 si credentials invalides
    """
    try:
        # Normaliser username (lowercase, strip)
        username = username.lower().strip()

        # Vérifier que l'utilisateur existe
        if not is_allowed_user(username):
            logger.warning(f"Login attempt for unknown user: {username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )

        # Récupérer les infos utilisateur
        user_info = get_user_info(username)
        if not user_info:
            logger.warning(f"User info not found for: {username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )

        # Vérifier le status
        if user_info.get("status") != "active":
            logger.warning(f"Login attempt for inactive user: {username}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account is inactive"
            )

        # Vérifier le password
        password_hash = user_info.get("password_hash")
        if not password_hash:
            logger.error(f"No password hash configured for user: {username}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication not configured for this user"
            )

        if not verify_password(password, password_hash):
            logger.warning(f"Invalid password for user: {username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )

        # Créer le JWT token
        token_data = {
            "sub": username,  # Subject = user_id
            "roles": user_info.get("roles", []),
            "label": user_info.get("label", username)
        }

        access_token = create_access_token(token_data)

        # Log succès pour audit
        logger.info(f"Successful login for user: {username}")

        # Retourner le token + user info
        return success_response({
            "token": access_token,
            "token_type": "bearer",
            "expires_in": ACCESS_TOKEN_EXPIRE_DAYS * 24 * 60 * 60,  # Secondes
            "user": {
                "id": user_info.get("id"),
                "label": user_info.get("label"),
                "roles": user_info.get("roles", [])
            }
        })

    except HTTPException:
        # Re-raise HTTPException as-is
        raise
    except Exception as e:
        logger.error(f"Unexpected error during login: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post("/logout")
async def logout():
    """
    Endpoint de logout (principalement côté client).

    Le client doit supprimer le token de localStorage.
    Pas besoin de blacklist côté serveur car tokens expirent automatiquement.

    Returns:
        {"ok": true, "message": "Logged out successfully"}
    """
    logger.info("Logout endpoint called (client-side token deletion)")
    return success_response({"message": "Logged out successfully"})


@router.get("/verify")
async def verify_token(token: str):
    """
    Endpoint pour vérifier si un token est valide.

    Utile pour le frontend pour vérifier l'expiration du token.

    Args:
        token: JWT token à vérifier (query param)

    Returns:
        {
            "ok": true,
            "data": {
                "valid": true,
                "user_id": "jack",
                "roles": ["admin"],
                "expires_at": "2025-01-04T12:00:00"
            }
        }
    """
    payload = decode_access_token(token)

    if not payload:
        return error_response("Invalid or expired token", code=401)

    # Extraire les infos du payload
    user_id = payload.get("sub")
    roles = payload.get("roles", [])
    exp_timestamp = payload.get("exp")

    expires_at = None
    if exp_timestamp:
        expires_at = datetime.utcfromtimestamp(exp_timestamp).isoformat()

    return success_response({
        "valid": True,
        "user_id": user_id,
        "roles": roles,
        "expires_at": expires_at
    })


@router.post("/change-password")
async def change_password(
    current_password: str = Form(...),
    new_password: str = Form(...),
    x_user: str = Header(None, alias="X-User")
):
    """
    Change le password de l'utilisateur actuel.

    Endpoint accessible à tous les users authentifiés pour changer leur propre password.

    Args:
        current_password: Password actuel (form data)
        new_password: Nouveau password (form data)
        x_user: User ID depuis header X-User

    Returns:
        {"ok": true, "message": "Password updated successfully"}

    Raises:
        HTTPException: 401 si current password incorrect, 400 si validation échoue
    """
    try:
        # Obtenir le user_id depuis X-User header
        from api.config.users import get_default_user, validate_user_id

        if not x_user:
            user_id = get_default_user()
        else:
            user_id = validate_user_id(x_user)

        # Récupérer user info
        user_info = get_user_info(user_id)
        if not user_info:
            logger.warning(f"Change password attempt for unknown user: {user_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        # Vérifier le current password
        current_hash = user_info.get("password_hash")
        if not current_hash:
            logger.error(f"No password hash for user: {user_id}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Password not configured for this user"
            )

        if not verify_password(current_password, current_hash):
            logger.warning(f"Invalid current password for user: {user_id}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Current password is incorrect"
            )

        # Valider nouveau password (min 8 caractères)
        if len(new_password) < 8:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="New password must be at least 8 characters"
            )

        # Hasher le nouveau password
        new_hash = get_password_hash(new_password)

        # Mettre à jour users.json
        import json
        from pathlib import Path

        users_path = Path(__file__).parent.parent / "config" / "users.json"
        with open(users_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # Trouver et mettre à jour l'utilisateur
        for user in config.get("users", []):
            if user.get("id") == user_id:
                user["password_hash"] = new_hash
                break

        # Sauvegarder
        with open(users_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        logger.info(f"Password changed successfully for user: {user_id}")

        return success_response({
            "message": "Password updated successfully"
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error changing password: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )
