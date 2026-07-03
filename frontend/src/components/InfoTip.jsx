import React, { useRef, useState } from 'react';
import ReactDOM from 'react-dom';

// Small (i) icon that explains, in plain words, what a feature helps an
// investor do. Hover on desktop, tap on mobile.
//
// The bubble is rendered in a portal with fixed positioning so it can never be
// clipped by parents with overflow-hidden (the premium cards all use it).
//
// Props:
//   title  - short heading inside the bubble (default "What is this for?")
//   dark   - set true when the icon sits on a dark panel (e.g. Terminal)
//   align  - kept for API compatibility; position is auto-clamped to the viewport
const BUBBLE_WIDTH = 288; // w-72

const InfoTip = ({ title = 'What is this for?', dark = false, align, children }) => {
    const [pos, setPos] = useState(null);
    const btnRef = useRef(null);

    const show = () => {
        if (!btnRef.current) return;
        const r = btnRef.current.getBoundingClientRect();
        const half = BUBBLE_WIDTH / 2;
        const left = Math.min(Math.max(r.left + r.width / 2, half + 8), window.innerWidth - half - 8);
        setPos({ top: r.bottom + 8, left });
    };
    const hide = () => setPos(null);

    return (
        <span className="relative inline-flex align-middle">
            <button
                ref={btnRef}
                type="button"
                aria-label={`Help: ${title}`}
                aria-expanded={!!pos}
                onClick={() => (pos ? hide() : show())}
                onBlur={hide}
                onMouseEnter={show}
                onMouseLeave={hide}
                className={`w-5 h-5 shrink-0 rounded-full flex items-center justify-center text-[11px] font-serif font-bold italic transition-colors cursor-help
                    ${dark
                        ? 'bg-white/10 text-gray-300 hover:bg-gold hover:text-navy'
                        : 'bg-navy/5 text-navy/50 hover:bg-gold hover:text-white'}`}
            >
                i
            </button>
            {pos && ReactDOM.createPortal(
                <span
                    role="tooltip"
                    style={{ position: 'fixed', top: pos.top, left: pos.left, transform: 'translateX(-50%)', width: BUBBLE_WIDTH, zIndex: 9999 }}
                    className="bg-navy text-cream text-xs leading-relaxed rounded-xl shadow-2xl p-4 normal-case font-normal tracking-normal text-left whitespace-normal pointer-events-none"
                >
                    <span className="block text-gold text-[10px] font-bold uppercase tracking-widest mb-1.5">{title}</span>
                    {children}
                </span>,
                document.body
            )}
        </span>
    );
};

export default InfoTip;
