"""
Configuration utilisateurs - RÉEXPORT pour backward compatibility.

NOTE: La logique est maintenant dans config/users.py
Ce fichier réexporte pour ne pas casser les imports existants.
"""
from config.users import (
    UserConfig,
    UsersDatabase,
    get_default_user,
    get_all_users,
    is_allowed_user,
    get_user_info,
    get_user_mode,
    clear_users_cache,
    validate_user_id,
)

__all__ = [
    "UserConfig",
    "UsersDatabase",
    "get_default_user",
    "get_all_users",
    "is_allowed_user",
    "get_user_info",
    "get_user_mode",
    "clear_users_cache",
    "validate_user_id",
]
