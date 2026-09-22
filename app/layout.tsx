import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "Multi-Agentic",
  description: "A supervised team of agents that researches, verifies and writes.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="min-h-screen flex flex-col">
          <header
            className="border-b sticky top-0 z-20"
            style={{ borderColor: "var(--line)", background: "var(--panel)" }}
          >
            <div className="max-w-[1400px] mx-auto px-4 h-12 flex items-center gap-4">
              <Link href="/" className="flex items-center gap-2 font-semibold text-[15px]">
                <span
                  className="inline-block w-2.5 h-2.5 rounded-full"
                  style={{ background: "var(--accent)" }}
                />
                Multi-Agentic
              </Link>
              <nav className="flex items-center gap-3 text-[13px]" style={{ color: "var(--muted)" }}>
                <Link href="/" className="hover:underline">Runs</Link>
                <Link href="/team" className="hover:underline">The team</Link>
                <Link href="/settings" className="hover:underline">Settings</Link>
              </nav>
            </div>
          </header>
          <main className="flex-1 max-w-[1400px] w-full mx-auto px-4 py-5">{children}</main>
        </div>
      </body>
    </html>
  );
}
