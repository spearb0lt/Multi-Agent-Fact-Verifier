"use client";

import { useSearchParams } from "next/navigation";
import { Suspense } from "react";
import Link from "next/link";
import RunView from "./RunView";

function Inner() {
  const key = useSearchParams().get("key") || "";
  if (!key) {
    return (
      <div className="panel p-5 max-w-lg mx-auto mt-10 text-center">
        <p className="text-sm mb-3" style={{ color: "var(--muted)" }}>
          No run was named in the address.
        </p>
        <Link href="/" className="btn">Back to runs</Link>
      </div>
    );
  }
  return <RunView runKey={key} />;
}

export default function RunPage() {
  // useSearchParams needs a Suspense boundary during prerender.
  return (
    <Suspense fallback={<p className="py-16 text-center text-sm" style={{ color: "var(--muted)" }}>Loading...</p>}>
      <Inner />
    </Suspense>
  );
}
