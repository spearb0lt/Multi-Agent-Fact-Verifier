import Link from "next/link";

export default function NotFound() {
  return (
    <div className="panel p-6 max-w-lg mx-auto mt-12 text-center">
      <h2 className="font-semibold mb-2">Not found</h2>
      <Link href="/" className="btn mt-2 inline-flex">
        Back to runs
      </Link>
    </div>
  );
}
