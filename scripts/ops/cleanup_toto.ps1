# Script pour supprimer complètement l'utilisateur "toto" (hard delete)
# Usage: .\scripts\ops\cleanup_toto.ps1

Write-Host "🗑️ Suppression permanente de l'utilisateur 'toto'..." -ForegroundColor Yellow

# Hard delete via API (nécessite que le serveur soit lancé)
$response = curl.exe -X DELETE "http://localhost:8080/admin/users/toto?hard_delete=true" `
    -H "X-User: jack" `
    -H "Content-Type: application/json"

Write-Host "✅ Réponse API:" -ForegroundColor Green
$response | ConvertFrom-Json | ConvertTo-Json -Depth 5

Write-Host ""
Write-Host "✅ 'toto' supprimé complètement. Vous pouvez maintenant le recréer." -ForegroundColor Green
