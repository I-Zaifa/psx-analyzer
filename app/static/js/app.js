/* ── PSX Analyzer – Client-side JavaScript ─────────────────────────────────── */

document.addEventListener('DOMContentLoaded', function() {
    // Mobile nav toggle
    const navToggle = document.getElementById('navToggle');
    const navLinks = document.getElementById('navLinks');
    if (navToggle && navLinks) {
        navToggle.addEventListener('click', function() {
            navLinks.classList.toggle('open');
        });
        // Close on outside click
        document.addEventListener('click', function(e) {
            if (!navToggle.contains(e.target) && !navLinks.contains(e.target)) {
                navLinks.classList.remove('open');
            }
        });
    }

    // Auto-resize plotly charts on window resize
    window.addEventListener('resize', function() {
        document.querySelectorAll('.js-plotly-plot').forEach(function(plot) {
            Plotly.Plots.resize(plot);
        });
    });
});

/* ── Utility Functions ─────────────────────────────────────────────────────── */

function formatPercent(val) {
    if (val === null || val === undefined || isNaN(val)) return 'N/A';
    return (val * 100).toFixed(2) + '%';
}

function formatNumber(val, decimals) {
    decimals = decimals !== undefined ? decimals : 2;
    if (val === null || val === undefined || isNaN(val)) return 'N/A';
    if (Math.abs(val) >= 1e9) return (val / 1e9).toFixed(decimals) + 'B';
    if (Math.abs(val) >= 1e6) return (val / 1e6).toFixed(decimals) + 'M';
    if (Math.abs(val) >= 1e3) return (val / 1e3).toFixed(decimals) + 'K';
    return val.toFixed(decimals);
}

function colorClass(val) {
    if (val === null || val === undefined || isNaN(val)) return '';
    return val > 0 ? 'cell-positive' : val < 0 ? 'cell-negative' : '';
}
