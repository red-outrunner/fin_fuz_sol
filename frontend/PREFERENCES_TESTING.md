# User Preferences & Caching System - Testing Guide

## Overview

The Ubomvu platform now includes a comprehensive user preferences system that:
- ✅ Saves user settings to localStorage immediately
- ✅ Loads data instantly on page load (before consent decision)
- ✅ Respects user privacy with clear consent
- ✅ Persists watchlist, theme, and all settings across sessions
- ✅ Works even if user declines consent (uses existing data)

## What Gets Saved

### 1. Watchlist
- **Storage Key**: `ubomvu_watchlist`
- **Format**: JSON array of ticker strings
- **Example**: `["SBK.JO", "NPN.JO", "AGL.JO"]`
- **Max Size**: 20 tickers

### 2. User Preferences
- **Storage Key**: `ubomvu_preferences`
- **Format**: JSON object
- **Saved Settings**:
  - `heatmapTimePeriod` - Selected time period (1d, 7d, 1mo, ytd)
  - `exportQuality` - Export preset (social, print, custom)
  - `dashboardLayout` - Dashboard layout preference
  - `screenerFilters` - Custom screener filter settings

### 3. Theme
- **Storage Key**: `ubomvu_theme`
- **Format**: String ("light" or "dark")
- **Example**: `"dark"`

### 4. Consent
- **Storage Key**: `ubomvu_consent` (cookie)
- **Format**: Cookie value
- **Values**: "accepted" (1 year) or "declined" (30 days)
- **Shown Flag**: `ubomvu_consent_shown` (localStorage)

## How to Test

### Test 1: Initial Setup
1. Open the application in a private/incognito window
2. You should see the consent toast at bottom-left
3. Click "Accept"
4. Add 3-5 stocks to your watchlist
5. Change heatmap time period to "7-Day"
6. Change export quality to "Print"
7. Switch to dark mode

**Expected Result**: All settings saved to localStorage

### Test 2: Page Reload
1. Refresh the page (F5 or Ctrl+R)
2. **Immediately check**:
   - Watchlist should show your stocks (no flicker)
   - Theme should be dark (no flash of light mode)
   - Heatmap should show 7-Day period selected
   - Export quality should show "Print" selected

**Expected Result**: All settings restored instantly

### Test 3: Browser Restart
1. Close the browser completely
2. Reopen browser and navigate to app
3. Check all settings are preserved

**Expected Result**: All settings still present

### Test 4: Multiple Sessions
1. Open app in Tab 1, add stocks to watchlist
2. Open app in Tab 2 (same domain)
3. Both tabs should show same watchlist

**Expected Result**: Settings sync across tabs

### Test 5: Decline Consent
1. Clear all data (see below)
2. Open app, click "Decline" on consent toast
3. Add stocks to watchlist
4. Refresh page

**Expected Result**: 
- Watchlist NOT saved (resets on refresh)
- App still functional
- Consent toast doesn't show again for 30 days

### Test 6: Change Mind
1. Accept consent initially
2. Use app, save settings
3. Clear data manually
4. Revisit app, decline consent
5. Check data is cleared

**Expected Result**: All localStorage data removed

## Manual Data Inspection

### Check localStorage
Open browser DevTools Console and run:

```javascript
// View all Ubomvu data
console.log('Watchlist:', JSON.parse(localStorage.getItem('ubomvu_watchlist')));
console.log('Preferences:', JSON.parse(localStorage.getItem('ubomvu_preferences')));
console.log('Theme:', localStorage.getItem('ubomvu_theme'));
console.log('Consent Shown:', localStorage.getItem('ubomvu_consent_shown'));
```

### Check Cookies
```javascript
// View consent cookie
console.log('Consent Cookie:', document.cookie.match(/ubomvu_consent=([^;]+)/));
```

### Clear All Data
```javascript
// Remove all Ubomvu data
['ubomvu_watchlist', 'ubomvu_preferences', 'ubomvu_theme', 'ubomvu_consent_shown']
  .forEach(key => localStorage.removeItem(key));
document.cookie = 'ubomvu_consent=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/';
console.log('All Ubomvu data cleared!');
```

## Debug Panel

A debug panel is available to visualize the current state:

1. Import and add `<PreferencesDebug />` to your App.jsx temporarily
2. It shows:
   - Consent status
   - Data loaded status
   - Current watchlist
   - Saved preferences
   - Storage info

## Performance Benchmarks

### Expected Load Times
- **First Visit**: < 100ms (no saved data)
- **Returning User**: < 50ms (data loaded from localStorage)
- **Settings Save**: Instant (< 10ms)
- **Watchlist Update**: Instant (< 10ms)

### No Network Requests
All preference data is stored locally - no API calls needed for:
- Loading watchlist
- Saving settings
- Theme switching
- Consent management

## Troubleshooting

### Settings Not Saving
1. Check browser console for errors
2. Verify localStorage is available (not in private mode)
3. Check if consent was given
4. Verify `updatePreference()` or `updateWatchlist()` is called

### Settings Not Loading
1. Check if data exists in localStorage
2. Verify UserPreferencesProvider wraps the component
3. Check if `loaded` state is true
4. Verify component uses `useUserPreferences()` hook

### Consent Toast Not Showing
1. Check cookie `ubomvu_consent`
2. Check localStorage `ubomvu_consent_shown`
3. Clear both to reset

### Watchlist Not Syncing
1. Verify you're using `updateWatchlist()` not direct setState
2. Check if watchlist array is being modified (use new array)
3. Verify max 20 tickers limit

## Production Checklist

Before deploying to production:

- [ ] Remove `<PreferencesDebug />` component
- [ ] Remove `test_preferences.js` file
- [ ] Test in production environment
- [ ] Verify GDPR/privacy compliance
- [ ] Update privacy policy if needed
- [ ] Test on mobile devices
- [ ] Test in different browsers (Chrome, Firefox, Safari, Edge)

## Success Criteria

✅ User adds stocks → refreshes → stocks still there
✅ User changes theme → closes browser → reopens → theme preserved
✅ User sets heatmap period → navigates away → returns → period saved
✅ User accepts consent → data saved immediately
✅ User declines consent → app works but doesn't save new data
✅ Returning user → all settings load before first render
✅ No flicker or loading state for saved preferences
✅ Works offline (no network required for preferences)

## Support

If you encounter issues:
1. Check browser console for errors
2. Verify localStorage is enabled
3. Try in different browser
4. Clear all Ubomvu data and retry
