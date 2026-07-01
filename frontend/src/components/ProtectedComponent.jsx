import React from 'react';

// Tiers/paywall removed — every feature is open to everyone. This is now a passthrough
// that renders its children regardless of the (now-ignored) tier props still passed by
// callers.
const ProtectedComponent = ({ children }) => children;

export default ProtectedComponent;
