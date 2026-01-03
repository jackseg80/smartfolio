# Fix Saxo OAuth2 Redirect URI Error

## Error
```json
{"error":"invalid_request","error_description":"Value of redirect_uri parameter is not registered","error_uri":null}
```

## Cause
The redirect URI `http://192.168.1.200:8080/api/saxo/callback` is not registered in your Saxo Developer Portal application settings.

## Solution

### 1. Login to Saxo Developer Portal
Visit: https://www.developer.saxo/openapi/appmanagement

### 2. Find Your Application
- Click on your application (the one with the Client ID from your .env)
- For **SIM environment**: use the app with `SAXO_SIM_CLIENT_ID`
- For **LIVE environment**: use the app with `SAXO_LIVE_CLIENT_ID`

### 3. Add Redirect URI
In the application settings, find the **Redirect URIs** section and add:

```
http://192.168.1.200:8080/api/saxo/callback
```

**Important:**
- Make sure there are NO trailing slashes: `/callback` (not `/callback/`)
- Use `http://` (not `https://`) for LAN development
- The IP must match your server IP exactly

### 4. Save Changes
Click "Save" or "Update" to register the redirect URI.

### 5. Test Again
- Wait 1-2 minutes for changes to propagate
- Go back to SmartFolio dashboard
- Click "Se connecter à Saxo"
- The OAuth flow should now work

---

## Multiple Environments (Optional)

If you want to access the dashboard from multiple locations, add all redirect URIs:

```
http://192.168.1.200:8080/api/saxo/callback
http://localhost:8080/api/saxo/callback
http://127.0.0.1:8080/api/saxo/callback
```

Saxo allows multiple redirect URIs per application.

---

## Still Not Working?

### Check .env Configuration
Verify your `.env` on the server:

```bash
cat ~/smartfolio/.env | grep SAXO
```

Should show:
```
SAXO_REDIRECT_URI=http://192.168.1.200:8080/api/saxo/callback
SAXO_ENVIRONMENT=live  # or sim
SAXO_LIVE_CLIENT_ID=fbd88bceea724e45b48f7d78a3730dec
SAXO_LIVE_CLIENT_SECRET=ab930bc61ce74269b75ef7d678e9ef94
```

### Check Application Matches Environment
- If `SAXO_ENVIRONMENT=sim`, use the SIM app (with SIM credentials)
- If `SAXO_ENVIRONMENT=live`, use the LIVE app (with LIVE credentials)

The redirect URI must be registered in the **correct app** for the active environment.
