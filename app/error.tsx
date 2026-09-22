"use client";

export default function Error({ error, reset }: { error: Error; reset: () => void }) {
  return (
    <div className="panel p-6 max-w-lg mx-auto mt-12">
      <h2 className="font-semibold mb-2">Something went wrong</h2>
      <p className="text-sm mb-4" style={{ color: "var(--muted)" }}>
        {error.message}
      </p>
      <button className="btn" onClick={reset}>
        Try again
      </button>
    </div>
  );
}
