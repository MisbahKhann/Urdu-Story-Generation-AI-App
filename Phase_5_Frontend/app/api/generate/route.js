/**
 * app/api/generate/route.js
 *
 * Next.js Route Handler that proxies POST /api/generate to the
 * FastAPI backend.  Keeping the backend URL server-side avoids
 * CORS issues and hides the raw backend URL from the browser.
 *
 * Set BACKEND_URL in your Vercel environment variables:
 *   BACKEND_URL=https://your-backend.onrender.com
 */

import { NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";

export async function POST(request) {
  try {
    const body = await request.json();

    const backendRes = await fetch(`${BACKEND_URL}/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!backendRes.ok) {
      const err = await backendRes.json().catch(() => ({ detail: "Backend error" }));
      return NextResponse.json(err, { status: backendRes.status });
    }

    const data = await backendRes.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error("Proxy error:", error);
    return NextResponse.json(
      { detail: "Failed to reach the story generation service." },
      { status: 502 }
    );
  }
}