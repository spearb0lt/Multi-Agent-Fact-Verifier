"use client";

/**
 * Keys and connection settings.
 *
 * A key typed here stays in this browser's local storage and rides only this
 * visitor's own requests. It is never written to the database, which is why a
 * run resumed after the server restarts falls back to the server's own keys.
 * That is a real limitation rather than a detail to leave the reader to find
 * out, so it is stated on the page.
 */

import { useEffect, useState } from "react";
import type { Config, ProviderStatus } from "@/lib/api";
import { api, setStoredKeys, setStoredPassword, storedKeys, storedPassword } from "@/lib/api";

export default function SettingsPage() {
  const [config, setConfig] = useState<Config | null>(null);
  const [keys, setKeys] = useState<Record<string, string>>({});
  const [password, setPassword] = useState("");
  const [saved, setSaved] = useState("");
  const [checking, setChecking] = useState(false);
  const [checked, setChecked] = useState<ProviderStatus[] | null>(null);

  useEffect(() => {
    setKeys(storedKeys());
    setPassword(storedPassword());
    api.config().then(setConfig).catch(() => undefined);
  }, []);

  const save = () => {
    const cleaned = Object.fromEntries(
      Object.entries(keys).filter(([, value]) => value.trim()),
    );
    setStoredKeys(cleaned);
    setStoredPassword(password.trim());
    setSaved("Saved in this browser.");
    setTimeout(() => setSaved(""), 2500);
  };

  const verify = async () => {
    setChecking(true);
    try {
      const result = await api.verifyKeys(
        Object.fromEntries(Object.entries(keys).filter(([, v]) => v.trim())),
      );
      setChecked(result.providers);
    } catch {
      setChecked(null);
    } finally {
      setChecking(false);
    }
  };

  if (!config) {
    return <p className="py-16 text-center text-sm" style={{ color: "var(--muted)" }}>Loading...</p>;
  }

  const status = checked || config.providers;

  return (
    <div className="space-y-5 max-w-3xl">
      <div>
        <h1 className="text-xl font-semibold mb-0.5">Settings</h1>
        <p className="text-[13px]" style={{ color: "var(--muted)" }}>
          Keys entered here stay in this browser and are sent only with your own
          requests. They are never stored on the server.
        </p>
      </div>

      {!config.allow_client_keys && (
        <p className="panel p-3 text-[13px]" style={{ color: "var(--warn)" }}>
          This deployment does not accept pasted keys. It uses the server&apos;s own.
        </p>
      )}

      <section className="panel p-4 space-y-3">
        <h2 className="text-xs font-semibold uppercase tracking-wide" style={{ color: "var(--muted)" }}>
          Provider keys
        </h2>
        <p className="text-[11px]" style={{ color: "var(--faint)" }}>
          A run started with a pasted key holds it in memory for as long as it runs.
          If the server restarts mid run, resuming falls back to the server&apos;s keys.
        </p>

        <div className="space-y-2">
          {status.map((provider) => (
            <div key={provider.id} className="flex items-center gap-2">
              <div className="w-32 shrink-0">
                <span className="text-[13px]">{provider.label}</span>
                <span
                  className="block text-[10px]"
                  style={{ color: provider.available ? "var(--ok)" : "var(--faint)" }}
                >
                  {provider.available ? "available" : provider.local ? "local server" : "no key"}
                </span>
              </div>
              <input
                className="field text-[12px] py-1.5 font-mono"
                type="password"
                placeholder={provider.local ? "not needed" : provider.key_names[0] || "API key"}
                value={keys[provider.id] || ""}
                disabled={provider.local}
                onChange={(e) => setKeys((k) => ({ ...k, [provider.id]: e.target.value }))}
              />
            </div>
          ))}
        </div>

        <div className="flex items-center gap-2">
          <button className="btn btn-primary" onClick={save}>Save</button>
          <button className="btn" onClick={verify} disabled={checking}>
            {checking ? "Checking..." : "Check the keys work"}
          </button>
          {saved && (
            <span className="text-[12px]" style={{ color: "var(--ok)" }}>{saved}</span>
          )}
        </div>
      </section>

      <section className="panel p-4 space-y-2">
        <h2 className="text-xs font-semibold uppercase tracking-wide" style={{ color: "var(--muted)" }}>
          Deployment password
        </h2>
        <p className="text-[11px]" style={{ color: "var(--faint)" }}>
          Only needed if this deployment sets APP_PASSWORD. With a password set, the
          live trace falls back to polling, because a browser cannot attach a header
          to an event stream.
        </p>
        <input
          className="field text-[13px] py-1.5"
          type="password"
          placeholder="X-App-Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
      </section>

      <section className="panel p-4 space-y-1.5 text-[12px]">
        <h2 className="text-xs font-semibold uppercase tracking-wide mb-1" style={{ color: "var(--muted)" }}>
          This deployment
        </h2>
        <Row label="Runtime" value={`${config.runtime.tier}, ${config.runtime.persistent_disk ? "durable disk" : "ephemeral disk"}`} />
        <Row label="Search backends" value={config.search.backends.join(", ") || "none"} />
        <Row label="Keyless search" value={config.search.keyless.join(", ") || "off"} />
        <Row label="Embeddings" value={config.embedding.available ? config.embedding.id : config.embedding.reason || "unavailable"} />
        <Row label="Concurrent runs" value={`${config.worker.active} of ${config.worker.capacity}`} />
        <Row
          label="Default ceilings"
          value={`${config.defaults.max_steps} steps, ${Math.round(config.defaults.max_tokens / 1000)}k tokens, $${config.defaults.max_usd}, ${Math.round(config.defaults.max_seconds / 60)} min`}
        />
      </section>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-4">
      <span style={{ color: "var(--faint)" }}>{label}</span>
      <span className="font-mono text-right break-all" style={{ color: "var(--muted)" }}>{value}</span>
    </div>
  );
}
