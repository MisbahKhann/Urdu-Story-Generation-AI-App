"use client";

import { useState, useRef, useEffect, useCallback } from "react";

// ── Token rendering helpers ───────────────────────────────────────────────────

const SPECIAL_TOKENS = new Set(["<EOS>", "<EOP>", "<EOT>"]);

/**
 * Split raw generated_text (space-separated tokens) into display segments.
 * Special tokens become badge components; regular tokens become plain text
 * with a space appended.
 */
function parseTokens(text) {
  if (!text) return [];
  return text.split(" ").filter(Boolean).map((tok, i) => ({
    id: i,
    raw: tok,
    isSpecial: SPECIAL_TOKENS.has(tok),
  }));
}

// ── Token component (animated) ───────────────────────────────────────────────

function Token({ token, isNew }) {
  if (token.isSpecial) {
    return (
      <span className={`special-token ${isNew ? "token-new" : ""}`}>
        {token.raw}
      </span>
    );
  }
  return (
    <span className={`token ${isNew ? "token-new" : ""}`}>
      {token.raw}{" "}
    </span>
  );
}

// ── Suggested prefixes ────────────────────────────────────────────────────────

const SUGGESTIONS = [
  "ایک دن بچہ باہر گیا",
  "پرانے وقتوں کی بات ہے",
  "بادشاہ نے کہا",
  "جنگل میں ایک شیر",
  "استاد نے بچوں سے کہا",
];

// ── Main component ────────────────────────────────────────────────────────────

export default function Home() {
  const [prefix, setPrefix]           = useState("");
  const [temperature, setTemperature] = useState(0.4);
  const [topK, setTopK]               = useState(5);
  const [maxLength, setMaxLength]     = useState(200);

  const [displayedTokens, setDisplayedTokens] = useState([]);
  const [newTokenIds, setNewTokenIds]          = useState(new Set());
  const [isGenerating, setIsGenerating]        = useState(false);
  const [isDone, setIsDone]                    = useState(false);
  const [stats, setStats]                      = useState(null);   // { token_count, stopped_at_eot }
  const [error, setError]                      = useState("");
  const [copied, setCopied]                    = useState(false);

  const storyBoxRef    = useRef(null);
  const revealTimerRef = useRef(null);

  // Auto-scroll story box as tokens appear
  useEffect(() => {
    if (storyBoxRef.current) {
      storyBoxRef.current.scrollTop = storyBoxRef.current.scrollHeight;
    }
  }, [displayedTokens]);

  // Cleanup timers on unmount
  useEffect(() => () => clearTimeout(revealTimerRef.current), []);

  // ── Step-wise reveal ──────────────────────────────────────────────────────

  const revealTokens = useCallback((allTokens) => {
    let index = 0;

    function revealNext() {
      if (index >= allTokens.length) {
        setIsGenerating(false);
        setIsDone(true);
        return;
      }

      const token = allTokens[index];
      setDisplayedTokens((prev) => [...prev, token]);
      setNewTokenIds(new Set([token.id]));

      // Remove "new" highlight after animation completes
      setTimeout(() => setNewTokenIds(new Set()), 350);

      index++;
      // Pace: faster for regular tokens, slight pause at sentence boundaries
      const delay = token.isSpecial ? 120 : 45;
      revealTimerRef.current = setTimeout(revealNext, delay);
    }

    revealNext();
  }, []);

  // ── Generate handler ──────────────────────────────────────────────────────

  const handleGenerate = useCallback(async () => {
    const trimmed = prefix.trim();
    if (!trimmed) return;

    // Reset state
    clearTimeout(revealTimerRef.current);
    setDisplayedTokens([]);
    setNewTokenIds(new Set());
    setIsDone(false);
    setError("");
    setStats(null);
    setIsGenerating(true);

    try {
      const res = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prefix: trimmed,
          max_length: maxLength,
          temperature,
          top_k: topK,
        }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: "Unknown error" }));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }

      const data = await res.json();
      const tokens = parseTokens(data.generated_text);
      setStats({ token_count: data.token_count, stopped_at_eot: data.stopped_at_eot });
      revealTokens(tokens);
    } catch (err) {
      setError(err.message);
      setIsGenerating(false);
    }
  }, [prefix, maxLength, temperature, topK, revealTokens]);

  // Handle Enter key (Shift+Enter = newline, Enter = generate)
  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!isGenerating) handleGenerate();
    }
  };

  // Copy story to clipboard
  const handleCopy = async () => {
    const text = displayedTokens
      .filter((t) => !t.isSpecial)
      .map((t) => t.raw)
      .join(" ");
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  };

  const hasOutput = displayedTokens.length > 0;

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <main className="app">
      {/* ── Header ── */}
      <header className="header">
        <span className="header-ornament">✦ ✦ ✦</span>
        <h1>داستان گو</h1>
        <p className="header-sub">Urdu Children&rsquo;s Story Generator</p>
        <div className="header-divider" />
      </header>

      {/* ── Card ── */}
      <div className="card">

        {/* Input label */}
        <label className="input-label" htmlFor="prefix-input">
          کہانی کا آغاز لکھیں
        </label>

        {/* Input row */}
        <div className="input-row">
          <textarea
            id="prefix-input"
            className="prefix-input"
            value={prefix}
            onChange={(e) => setPrefix(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="مثال: ایک دن بچہ باہر گیا…"
            rows={2}
            disabled={isGenerating}
          />
          <button
            className="generate-btn"
            onClick={handleGenerate}
            disabled={isGenerating || !prefix.trim()}
          >
            {isGenerating ? "لکھ رہا ہے…" : "لکھو"}
          </button>
        </div>

        {/* Suggestions */}
        <div style={{ marginTop: "0.75rem", display: "flex", gap: "0.5rem", flexWrap: "wrap", direction: "rtl" }}>
          {SUGGESTIONS.map((s) => (
            <button
              key={s}
              onClick={() => setPrefix(s)}
              disabled={isGenerating}
              style={{
                background: "transparent",
                border: "1px solid rgba(201,150,58,0.25)",
                color: "rgba(245,237,216,0.55)",
                borderRadius: "20px",
                padding: "0.25rem 0.75rem",
                fontSize: "0.75rem",
                cursor: "pointer",
                fontFamily: "var(--font-urdu)",
                transition: "all 0.2s",
              }}
              onMouseEnter={(e) => {
                e.target.style.borderColor = "rgba(201,150,58,0.6)";
                e.target.style.color = "rgba(245,237,216,0.9)";
              }}
              onMouseLeave={(e) => {
                e.target.style.borderColor = "rgba(201,150,58,0.25)";
                e.target.style.color = "rgba(245,237,216,0.55)";
              }}
            >
              {s}
            </button>
          ))}
        </div>

        {/* Settings sliders */}
        <div className="settings">
          <div className="setting-item">
            <label>تخلیقیت (Temperature)</label>
            <input
              type="range" min="0.1" max="1.5" step="0.1"
              value={temperature}
              onChange={(e) => setTemperature(parseFloat(e.target.value))}
              disabled={isGenerating}
            />
            <span className="value-display">{temperature.toFixed(1)}</span>
          </div>

          <div className="setting-item">
            <label>انتخاب (Top-K)</label>
            <input
              type="range" min="1" max="50" step="1"
              value={topK}
              onChange={(e) => setTopK(parseInt(e.target.value))}
              disabled={isGenerating}
            />
            <span className="value-display">{topK}</span>
          </div>

          <div className="setting-item">
            <label>لمبائی (Max Length)</label>
            <input
              type="range" min="30" max="400" step="10"
              value={maxLength}
              onChange={(e) => setMaxLength(parseInt(e.target.value))}
              disabled={isGenerating}
            />
            <span className="value-display">{maxLength}</span>
          </div>
        </div>

        {/* ── Story output ── */}
        {hasOutput && (
          <div className="story-section">
            <div className="story-header">
              <span className="story-title">✦ کہانی</span>
              {!isGenerating && (
                <button className="copy-btn" onClick={handleCopy}>
                  {copied ? "✓ Copied" : "Copy"}
                </button>
              )}
            </div>

            <div
              className="story-box"
              ref={storyBoxRef}
              style={{ maxHeight: "400px", overflowY: "auto" }}
            >
              <p className="story-text">
                {displayedTokens.map((token) => (
                  <Token
                    key={token.id}
                    token={token}
                    isNew={newTokenIds.has(token.id)}
                  />
                ))}
                {isGenerating && <span className="cursor" />}
              </p>
            </div>

            {/* Stats */}
            {stats && !isGenerating && (
              <div className="stats-bar">
                <span className="stat">tokens: <span>{stats.token_count}</span></span>
                <span className="stat">
                  ended: <span>{stats.stopped_at_eot ? "⟨EOT⟩" : "max length"}</span>
                </span>
                <span className="stat">
                  temp: <span>{temperature}</span> · top-k: <span>{topK}</span>
                </span>
              </div>
            )}
          </div>
        )}

        {/* Empty state */}
        {!hasOutput && !isGenerating && !error && (
          <div className="empty-state">
            <span className="empty-icon">📖</span>
            <p className="empty-text">کہانی کا آغاز لکھیں اور لکھو دبائیں</p>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="error-box">
            ⚠️ {error}
          </div>
        )}
      </div>

      {/* ── Footer ── */}
      <footer className="footer">
        CS 4063 · NLP Assignment · Urdu Story Generation System
      </footer>
    </main>
  );
}