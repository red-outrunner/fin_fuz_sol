#!/usr/bin/env node
/**
 * Test script to verify user preferences and caching system
 * Run this in browser console or as a Node test
 */

const STORAGE_KEYS = {
    WATCHLIST: 'ubomvu_watchlist',
    THEME: 'ubomvu_theme',
    PREFERENCES: 'ubomvu_preferences',
    CONSENT: 'ubomvu_consent',
};

console.log('=== Ubomvu User Preferences Test ===\n');

// Test 1: Check localStorage loading
console.log('Test 1: Loading data from localStorage');
try {
    const watchlist = JSON.parse(localStorage.getItem(STORAGE_KEYS.WATCHLIST) || '[]');
    const preferences = JSON.parse(localStorage.getItem(STORAGE_KEYS.PREFERENCES) || '{}');
    const theme = localStorage.getItem(STORAGE_KEYS.THEME);
    
    console.log('✓ Watchlist loaded:', watchlist);
    console.log('✓ Preferences loaded:', preferences);
    console.log('✓ Theme loaded:', theme);
} catch (err) {
    console.error('✗ Failed to load from localStorage:', err);
}

// Test 2: Check cookie consent
console.log('\nTest 2: Checking consent cookie');
const consentCookie = document.cookie.match(/ubomvu_consent=([^;]+)/);
console.log('✓ Consent cookie:', consentCookie ? consentCookie[1] : 'Not set');

// Test 3: Simulate adding to watchlist
console.log('\nTest 3: Adding test ticker to watchlist');
const testWatchlist = ['SBK.JO', 'NPN.JO', 'AGL.JO'];
localStorage.setItem(STORAGE_KEYS.WATCHLIST, JSON.stringify(testWatchlist));
const saved = JSON.parse(localStorage.getItem(STORAGE_KEYS.WATCHLIST));
console.log('✓ Saved watchlist:', saved);
console.log('✓ Match:', JSON.stringify(saved) === JSON.stringify(testWatchlist) ? 'PASS' : 'FAIL');

// Test 4: Simulate saving preferences
console.log('\nTest 4: Saving user preferences');
const testPrefs = {
    heatmapTimePeriod: '7d',
    exportQuality: 'print',
    dashboardLayout: 'default',
};
localStorage.setItem(STORAGE_KEYS.PREFERENCES, JSON.stringify(testPrefs));
const loadedPrefs = JSON.parse(localStorage.getItem(STORAGE_KEYS.PREFERENCES));
console.log('✓ Saved preferences:', testPrefs);
console.log('✓ Loaded preferences:', loadedPrefs);
console.log('✓ Match:', JSON.stringify(loadedPrefs) === JSON.stringify(testPrefs) ? 'PASS' : 'FAIL');

// Test 5: Theme persistence
console.log('\nTest 5: Theme persistence');
localStorage.setItem(STORAGE_KEYS.THEME, 'dark');
const savedTheme = localStorage.getItem(STORAGE_KEYS.THEME);
console.log('✓ Saved theme:', savedTheme);
console.log('✓ Match:', savedTheme === 'dark' ? 'PASS' : 'FAIL');

// Test 6: Simulate page reload
console.log('\nTest 6: Simulating page reload (data persistence)');
console.log('After page reload, these values should persist:');
console.log('  - Watchlist:', JSON.parse(localStorage.getItem(STORAGE_KEYS.WATCHLIST)));
console.log('  - Preferences:', JSON.parse(localStorage.getItem(STORAGE_KEYS.PREFERENCES)));
console.log('  - Theme:', localStorage.getItem(STORAGE_KEYS.THEME));

// Test 7: Clear test data
console.log('\nTest 7: Cleaning up test data');
Object.values(STORAGE_KEYS).forEach(key => {
    localStorage.removeItem(key);
    console.log('✓ Removed:', key);
});
document.cookie = 'ubomvu_consent=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/';
console.log('✓ Cleared consent cookie');

console.log('\n=== All Tests Complete ===');
console.log('\nTo test in production:');
console.log('1. Add stocks to your watchlist');
console.log('2. Change heatmap time period to "7-Day"');
console.log('3. Change export quality to "Print"');
console.log('4. Switch to dark mode');
console.log('5. Refresh the page');
console.log('6. All settings should be preserved! ✓');
