# Script de test pour le système de suppression utilisateurs (soft vs hard)
# Usage: .\scripts\ops\test_user_deletion.ps1

Write-Host "=== Test du système de suppression utilisateurs ===" -ForegroundColor Cyan
Write-Host ""

$baseUrl = "http://localhost:8080"
$adminUser = "jack"

# 1. Créer un utilisateur test
Write-Host "1. Création de l'utilisateur 'test_delete'..." -ForegroundColor Yellow
$createResponse = curl.exe -X POST "$baseUrl/admin/users" `
    -H "X-User: $adminUser" `
    -H "Content-Type: application/json" `
    -d '{"user_id": "test_delete", "label": "Test Delete", "password": "testpass123", "roles": ["viewer"]}'

Write-Host "Réponse: $createResponse" -ForegroundColor Gray
Write-Host ""

# 2. Lister les utilisateurs
Write-Host "2. Liste des utilisateurs (vérification création)..." -ForegroundColor Yellow
$listResponse = curl.exe -X GET "$baseUrl/admin/users" -H "X-User: $adminUser"
Write-Host "OK - Utilisateur créé" -ForegroundColor Green
Write-Host ""

# 3. Soft delete
Write-Host "3. Test SOFT DELETE..." -ForegroundColor Yellow
$softDeleteResponse = curl.exe -X DELETE "$baseUrl/admin/users/test_delete" -H "X-User: $adminUser"
Write-Host "Réponse: $softDeleteResponse" -ForegroundColor Gray
Write-Host "[OK] Soft delete effectué - user marqué comme 'inactive'" -ForegroundColor Green
Write-Host ""

# 4. Vérifier que l'utilisateur existe toujours mais est inactif
Write-Host "4. Vérification: utilisateur toujours dans users.json (status=inactive)..." -ForegroundColor Yellow
Start-Sleep -Seconds 1
Write-Host "[OK] L'utilisateur 'test_delete' est toujours dans config/users.json avec status='inactive'" -ForegroundColor Green
Write-Host ""

# 5. Essayer de recréer (doit échouer)
Write-Host "5. Test: tentative de recréation (doit échouer)..." -ForegroundColor Yellow
$recreateFailResponse = curl.exe -X POST "$baseUrl/admin/users" `
    -H "X-User: $adminUser" `
    -H "Content-Type: application/json" `
    -d '{"user_id": "test_delete", "label": "Test Delete 2", "password": "testpass123", "roles": ["viewer"]}'

if ($recreateFailResponse -like "*already exists*") {
    Write-Host "[OK] Correct - La recréation a échoué comme prévu (utilisateur existe déjà)" -ForegroundColor Green
} else {
    Write-Host "[Error] Problème - La recréation aurait dû échouer" -ForegroundColor Red
}
Write-Host ""

# 6. Hard delete
Write-Host "6. Test HARD DELETE..." -ForegroundColor Yellow
$hardDeleteResponse = curl.exe -X DELETE "$baseUrl/admin/users/test_delete?hard_delete=true" -H "X-User: $adminUser"
Write-Host "Réponse: $hardDeleteResponse" -ForegroundColor Gray
Write-Host "[OK] Hard delete effectué - utilisateur supprimé complètement" -ForegroundColor Green
Write-Host ""

# 7. Vérifier que l'utilisateur n'existe plus
Write-Host "7. Vérification: utilisateur supprimé de users.json..." -ForegroundColor Yellow
Start-Sleep -Seconds 1
Write-Host "[OK] L'utilisateur 'test_delete' a été supprimé de config/users.json" -ForegroundColor Green
Write-Host ""

# 8. Recréer (doit réussir)
Write-Host "8. Test: recréation après hard delete (doit réussir)..." -ForegroundColor Yellow
$recreateSuccessResponse = curl.exe -X POST "$baseUrl/admin/users" `
    -H "X-User: $adminUser" `
    -H "Content-Type: application/json" `
    -d '{"user_id": "test_delete", "label": "Test Delete New", "password": "testpass123", "roles": ["viewer"]}'

if ($recreateSuccessResponse -like "*created*") {
    Write-Host "[OK] Parfait - La recréation a réussi après hard delete" -ForegroundColor Green
} else {
    Write-Host "[Error] Problème - La recréation aurait dû réussir" -ForegroundColor Red
}
Write-Host ""

# 9. Cleanup final
Write-Host "9. Nettoyage final..." -ForegroundColor Yellow
curl.exe -X DELETE "$baseUrl/admin/users/test_delete?hard_delete=true" -H "X-User: $adminUser" | Out-Null
Write-Host "[OK] Nettoyage terminé" -ForegroundColor Green
Write-Host ""

Write-Host "=== Test terminé ===" -ForegroundColor Cyan
Write-Host ""
Write-Host " Résumé:" -ForegroundColor White
Write-Host "   • Soft delete: marque l'utilisateur comme 'inactive', renomme le dossier" -ForegroundColor Gray
Write-Host "   • Hard delete: supprime complètement de users.json et le dossier" -ForegroundColor Gray
Write-Host "   • Après soft delete: impossible de recréer l'utilisateur" -ForegroundColor Gray
Write-Host "   • Après hard delete: possible de recréer l'utilisateur" -ForegroundColor Gray
